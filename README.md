# Lung Nodule Segmentation with U-Net (LUNA16)

This project implements a deep learning pipeline for lung nodule segmentation using the LUNA16 dataset. The goal is to automatically detect and segment pulmonary nodules from CT scans, a crucial step in early lung cancer screening. A U-Net CNN architecture is used for the deep learning model.

---

## 1. Original CT volume

Each LUNA16 scan is a 3D volume of size roughly #slices × 512 × 512.

#slices (depth) varies per patient, e.g. 200–400 slices.

Each slice is 512×512 pixels in the XY plane

---

## 2. Annotation Mask

Radiologists provided the center coordinates (x,y,z) and a diameter for each nodule in the annotations.csv file.

You take each center, expand outward by nodule radius in 3D, and fill those voxels into a binary mask:

1 = nodule voxel  
0 = everything else

So you end up with a binary mask the same shape as the CT volume: #slices × 512 × 512

---

## 3. Training data

If you fed whole slices (512×512), there are few positives nodules compared to the background. This leads to a huge class imbalance. It would cause the model to predict 0 everywhere as it still gets a low loss by doing that.

To fix that, you extract localized crops. You extract a 7x96x96 chunk around the known nodule given its coordinates. We extract 3 slices above and 3 slices below and the center, making 7 slices. The other 6 slices give it spatial context. During training, we crop a random 7x64x64 patch to serve as a form of data augmentation by giving the nodule a random offset. This prevents the model from predicting the center always. The ground truth label is extracted from the Annotation Mask and it is a 1x64x64 chunk based on the center slice. 

In practice, a nodule is usually visible in just 1 or 2 consecutive slices. That’s why the ground truth mask for segmentation is only 1 slice thick. Remember, the ground truth label is extracted from the annotation mask we created in step 2.

---

## 4. Loss

Dice loss was utilized as the loss function. Recall was maximized over precision, since missing a nodule (false negative) is far worse than having some extra false alarms (false positives). In cancer screening, missing a nodule (false negative) is far worse than having some extra false alarms (false positives). Radiologists can review false positives, but a missed tumor might cost a life.

The loss function was reweighted so that false negatives were penalized heavily while false positives were lightly penalized. Penalizing false negatives heavily also helped prevent the model from blindly predicting all 0s (no nodule).

This results in the model producing many “hotspots” (high false positives). But recall is maximized, so the vast majority of actual nodules are caught.

---

## 5. Model Architecture

- U-Net CNN with **7-channel input** (multi-slice context).
- Input sizes:
  - Training: `7x64x64`
  - Validation: `7x512x512`
- Final **1 × 1 convolution** reduces to 1 channel.
- **Sigmoid activation** → threshold at `0.5` → binary output.
- Output sizes:
  - Training: `1 × 64 × 64`
  - Validation: `1 × 512 × 512`

---

## 6. Validation data

For validation, you don’t crop around nodules. You pass full slices: 7 × 512 × 512 around each slice index because we want to test to see how well the model does on the actual 512x512 CT Scan data.

Ground Truth Label = 1 × 512 × 512 binary mask for the center slice.

Basically, this checks if the model generalizes across the whole CT, not just cropped regions.

---

## 7. Model Output

At the very end of the U-Net:  
There’s a 1×1 convolution layer that reduces the multi-channel feature map down to 1 channel. Then, a Sigmoid function is applied at the end width > 0.5 to get the 0 and 1.

After passing in the 7x64x64 (training) or 7x512x512 (validation) chunks in the U-Net segmentation model, it will output the predicted 1x64x64 (training) or 1x512x512 (validation) chunks. We then compare these predicted chunks to their corresponding ground truth labels (based on the binary annotation mask).

---

## 8. Training Setup

Hardware: **NVIDIA A10 GPU ×1** 

Example command:

```bash
python train.py --epochs 10 --batch-size 16 --num-workers 8 --augmented
