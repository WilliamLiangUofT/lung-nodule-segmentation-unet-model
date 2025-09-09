import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

from model_unet import UNetWrapper
from augmentation_model import DataAugmentation
from datasets import Luna2DSegmentationDataset, TrainingLuna2DSegmentationDataset, getCT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_SIZE = 4
METRICS_LOSS_IND = 0
METRICS_TP_IND = 1
METRICS_FN_IND = 2
METRICS_FP_IND = 3


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None: # If it's not none, it means sys_argv was passed in as a list
            # sys.arv[1:] gets the parameters from the command line argmenst. 
            # Example python train.py --epochs=3 --num-workers=4 will be sys.argv == ['train.py', '--epochs=3', '--num-workers=4']
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', help='Batch size to use for training', default=16, type=int)
        parser.add_argument('--num-workers', help='Number of worker processes for background data loading', default=8, type=int,)
        parser.add_argument('--epochs', help='Number of epochs to train for', default=1, type=int,)
        parser.add_argument('--augmented', help="Augment the training data.", action='store_true', default=False,)

        self.cli_args = parser.parse_args(sys_argv)

        self.augmentation_dict = {}
        if self.cli_args.augmented:
            self.augmentation_dict.update({'flip': True, 'offset': 0.03,
                                           'scale': 0.2, 'rotate': True,
                                           'noise': 25.0})

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.writer_train = SummaryWriter(log_dir="runs/train")
        self.writer_val = SummaryWriter(log_dir="runs/val")

        self.unet_segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.best_recall_score = 0.0

    
    def init_model(self):
        unet_segmentation_model = UNetWrapper(input_channels=7, output_channels=1, depth=3, filters=4, padding=1, batch_norm=True)
        augmentation_model = DataAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))

            if torch.cuda.device_count() > 1: # If multiple GPUs, we will wrap our models around DataParallel which automatically lets our model train across multiple GPUs
                unet_segmentation_model = nn.DataParallel(unet_segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            
            # Transfers to GPU
            unet_segmentation_model = unet_segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)
        
        return unet_segmentation_model, augmentation_model

    
    def init_optimizer(self):
        return optim.Adam(self.unet_segmentation_model.parameters(), lr=1e-4, weight_decay=1e-6)


    def init_training_dataloader(self):
        training_dataset = TrainingLuna2DSegmentationDataset(val_stride=10, is_val_set=False, context_slices_count=3)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            # Each GPU will get batch_size / num_gpus samples.
            # This ensures each GPU is still computing the original batch_size, just multiple computing batches in parallel
            batch_size *= torch.cuda.device_count()
        
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)
        return training_dataloader


    def init_validation_dataloader(self):
        validation_dataset = Luna2DSegmentationDataset(val_stride=10, is_val_set=True, context_slices_count=3)
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)
        return validation_dataloader

    
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        training_dataloader = self.init_training_dataloader()
        validation_dataloader = self.init_validation_dataloader()

        for epoch in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch,
                self.cli_args.epochs,
                len(training_dataloader),
                len(validation_dataloader),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            train_metrics = self.do_training(epoch, training_dataloader)
            self.log_metrics(epoch, 'train', train_metrics)

            if epoch == 1 or epoch % 5 == 0: # Evaluate on the validation dataset at 1, 5, 10, 15, etc.
                val_metrics = self.do_validation(epoch, validation_dataloader)
                recall_score = self.log_metrics(epoch, 'val', val_metrics)
                if recall_score > self.best_recall_score:
                    self.best_recall_score = recall_score
                    self.save_model_state(epoch)
                
                self.log_images(epoch, 'train', training_dataloader)
                self.log_images(epoch, 'val', validation_dataloader)
        
        self.writer_train.close()
        self.writer_val.close()


    def do_training(self, epoch, training_dataloader):
        training_metrics = torch.zeros(METRICS_SIZE, len(training_dataloader.dataset), device=self.device)

        self.unet_segmentation_model.train()
        training_dataloader.dataset.shuffle_samples()

        for batch_index, batch_tuple in enumerate(training_dataloader):
            self.optimizer.zero_grad()
            loss = self.compute_batch_loss(batch_index, batch_tuple, training_dataloader.batch_size, training_metrics)
            loss.backward()
            self.optimizer.step()

        return training_metrics.to('cpu')
    

    def do_validation(self, epoch, validation_dataloader):
        with torch.no_grad():
            validation_metrics = torch.zeros(METRICS_SIZE, len(validation_dataloader.dataset), device=self.device)

            self.unet_segmentation_model.eval()

            for batch_index, batch_tuple in enumerate(validation_dataloader):
                self.compute_batch_loss(batch_index, batch_tuple, validation_dataloader.batch_size, validation_metrics)

        return validation_metrics.to('cpu')

    
    def compute_batch_loss(self, batch_index, batch_tuple, batch_size, metrics):
        input_s, label_s, _, _ = batch_tuple

        input_d = input_s.to(self.device, non_blocking=True)
        label_d = label_s.to(self.device, non_blocking=True)

        if self.unet_segmentation_model.training and self.augmentation_dict: # Validation should be augmenting data
            input_d, label_d = self.augmentation_model(input_d, label_d)
        
        prediction = self.unet_segmentation_model(input_d)
        dice_loss = self.dice_loss(prediction, label_d) # Total Loss across this entire single batch
        false_neg_loss = self.dice_loss(prediction * label_d, label_d) # Total Loss across this entire single batch. We do this to maximize recall
        
        label_mask = label_d.to(torch.float32)

        false_pos_map = prediction * (1.0 - label_mask)
        false_pos_loss = false_pos_map.mean()

        start_ind = batch_index * batch_size
        end_ind = start_ind + input_s.size(0)

        with torch.no_grad():
            prediction_bool_mask = (prediction[:, 0:1] > 0.5).to(torch.float32) # Converts the prediction sigmoid probabilities to either 0 or 1 based on threshold

            tp = (prediction_bool_mask * label_mask).sum(dim=[1, 2, 3]) # Sums along each single training example in the batch
            fn = ((1 - prediction_bool_mask) * label_mask).sum(dim=[1, 2, 3])
            fp = (prediction_bool_mask * (1 - label_mask)).sum(dim=[1, 2, 3])

            metrics[METRICS_LOSS_IND, start_ind:end_ind] = dice_loss
            metrics[METRICS_TP_IND, start_ind:end_ind] = tp
            metrics[METRICS_FN_IND, start_ind:end_ind] = fn
            metrics[METRICS_FP_IND, start_ind:end_ind] = fp

        return dice_loss.mean() + false_neg_loss.mean() * 8 + false_pos_loss * 0.1 # We do mean() to get average loss per training example in this batch

    
    def dice_loss(self, prediction, label, episilon=1):
        dice_prediction = prediction.sum(dim=[1, 2, 3]) # Returns something like [1 5 2 5], which is the sum for each training example in the single batch the number of 1s (True mask) predicted
        dice_label = label.sum(dim=[1, 2, 3]) # Same thing, but for the ground truth data
        dice_correct = (prediction * label).sum(dim=[1, 2, 3])

        dice_ratio = (2 * dice_correct + episilon) / (dice_prediction + dice_label + episilon)

        return 1 - dice_ratio


    def log_metrics(self, epoch, mode_str, metrics):
        log.info("E{} {}".format(epoch, type(self).__name__))

        metrics_n = metrics.detach().numpy()
        sum_n = metrics.sum(axis=1) # For each metric, you get the total sum of it across ALL examples in this epoch
        all_label_count = sum_n[METRICS_TP_IND] + sum_n[METRICS_FN_IND] # Total TP and FN for all training exampels (and each example is an image with many voxels. Each voxel will be TP, FN, etc)

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_n[METRICS_LOSS_IND].mean() # average loss of all training/validation examples
        
        # Percentages relative to all actual positive pixels in the label (TP + FN)
        metrics_dict['percent_all/tp'] = sum_n[METRICS_TP_IND] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_n[METRICS_FN_IND] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_n[METRICS_FP_IND] / (all_label_count or 1) * 100

        # Actual Precision, Recall, and F1 Scores
        metrics_dict['pr/precision'] = sum_n[METRICS_TP_IND] / ((sum_n[METRICS_TP_IND] + sum_n[METRICS_FP_IND]) or 1)
        metrics_dict['pr/recall'] = sum_n[METRICS_TP_IND] / ((sum_n[METRICS_TP_IND] + sum_n[METRICS_FN_IND]) or 1)
        metrics_dict['pr/f1_score'] = 2 * (metrics_dict['pr/precision'] * metrics_dict['pr/recall']) / ((metrics_dict['pr/precision'] + metrics_dict['pr/recall']) or 1)

        # Logging the Metrics Dictionary
        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch,
            mode_str + '_all',
            **metrics_dict,
        ))

        writer = getattr(self, f'writer_{mode_str}')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, epoch) # Epoch will be the x-axis in tensorboard graphs
        
        writer.flush()

        return metrics_dict['pr/recall'] # Recall is the most important to us. We want to MINIMIZE FALSE NEGATIVES. False Positives aren't a big issue, thus precision isn't as important


    def log_images(self, epoch, mode_str, dataloader):
        self.unet_segmentation_model.eval()

        series_ids = sorted(dataloader.dataset.series_list)[:12] # 12 CT Scan images' series-uid
        for series_index, series_uid in enumerate(series_ids):
            ct = getCT(series_uid)

            # 6 evenly spaced slices along the Index axis of the CT Scan
            for slice_slot in range(6):
                ct_index = slice_slot * (ct.ct_HU_voxels.shape[0] - 1) // 5 # One of the 6 indexes
                ct_t, label_t, _, _ = dataloader.dataset.getitem_full_ct_slice(series_uid, ct_index)

                input_g = ct_t.to(self.device).unsqueeze(0) # [1,7,512,512]
                label_g = label_t.to(self.device).unsqueeze(0) # [1,1,512,512]

                # Run the model and threshold the sigmoid output at 0.5
                pred_g = self.unet_segmentation_model(input_g)[0] # remove batch, now [1,512,512]
                pred_a = (pred_g.detach().cpu().numpy()[0] > 0.5) # 2D bool array
                lbl_a  = (label_g.cpu().numpy()[0][0] > 0.5) # 2D bool array

                # Normalize HU units
                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                center_slice = ct_t[dataloader.dataset.context_slices_count].numpy() # [1, 512, 512], the center slice

                # Build RGB Image. Starts with gray background. For the 3 channels, sets them to each be 512, 512, 1
                image = np.zeros((512, 512, 3), dtype=np.float32)
                image[:,:,:] = center_slice.reshape((512, 512, 1))

                # Now we add the colours
                # false positives  = pred & ~lbl → add to RED channel
                # false negatives  = ~pred & lbl → add to RED channel
                # true positives   = pred & lbl → add to GREEN channel (full intensity)
                image[:,:,0] += pred_a & (1 - lbl_a)
                image[:,:,0] += (1 - pred_a) & lbl_a
                image[:,:,1] += pred_a & lbl_a

                # Blend and clip to [0,1]. Tensorboard expects this range. 
                image *= 0.5
                image.clip(0, 1, image)

                # Write to tensorboard
                writer = getattr(self, f'writer_{mode_str}')
                writer.add_image(f'{mode_str}/series{series_index}_slice{slice_slot}', image, epoch, dataformats='HWC')

                if epoch == 1:
                    # TO log the ground truth mask in green. Pixels will be green for true positives (nodules)
                    label_only = np.zeros((512, 512, 3), dtype=np.float32)
                    label_only[:,:,:] = center_slice.reshape((512, 512, 1))
                    label_only[:,:,1] += lbl_a # Label_only is all 0 pixels. We then add 1 to green channel each pixel if the ground truth lbl_a had it. So, this lights green for TRUE POSITIVES

                    # Blend and clip again
                    label_only *= 0.5
                    label_only.clip(0,1,label_only)
                    writer.add_image(f'{mode_str}/series{series_index}_slice{slice_slot}_label', label_only, epoch, dataformats='HWC')
                
                writer.flush()
    

    def save_model_state(self, epoch):
        ckpt_dir = 'modelCheckpoints'
        os.makedirs(ckpt_dir, exist_ok=True) # Make directory if doesn't exist

        # Using DataParallel, unwrap the .module to get the real model
        model = self.unet_segmentation_model
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        filename = f"unet_epoch{epoch}.pth"
        filepath = os.path.join(ckpt_dir, filename)

        torch.save(model.state_dict(), filepath)
        log.info("Saved model parameters to {}".format(filepath))


# Starts the training when you run train.py in script. This tells it to call main() to start.
# Tells Python to run the main() method of SegmentationTrainingApp if this file is being run directly.
if __name__ == "__main__":
    SegmentationTrainingApp().main()


