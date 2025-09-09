import torch
import torch.nn.functional as F
from torch import nn

import math

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, depth, filters, padding=1, batch_norm=False):
        super().__init__()

        prev_channels = input_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (filters + i), padding, batch_norm))
            prev_channels = 2 ** (filters + i)


        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)): # Don't apply this in the last depth block
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (filters + i), padding, batch_norm))
            prev_channels = 2 ** (filters + i)
        
        self.last = nn.Conv2d(prev_channels, output_channels, kernel_size=1) # 1x1 Convolution to control number of final channels


    def forward(self, x):
        skip_blocks = []
        # Down Path
        for i, down_block in enumerate(self.down_path):
            x = down_block(x)
            if i != len(self.down_path) - 1: # In the U-shape, the last block doesn't get downsized by MaxPooling
                skip_blocks.append(x)
                x = F.max_pool2d(x, 2) # Could also not use functional API and define nn.MaxPool2d(2) in _init_()

        # Up Path
        for i, up_block in enumerate(self.up_path):
            x = up_block(x, skip_blocks[-i-1])
    
        x = self.last(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, batch_norm=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.relu = nn.ReLU()

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()


    def forward(self, x):
        x = self.bn(self.relu(self.conv1(x)))
        x = self.bn(self.relu(self.conv2(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, batch_norm=False):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_channels, out_channels, padding, batch_norm)

    
    # This is done to match the dimensions of the tranposed block and the skip connection block. It crops the skip block
    def crop_center(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()

        # These diffs are calculated to ensure we get a center crop
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2

        # Remember first 2 in the shape are batch and channel
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
    

    # Remember, tensor dimension for 2D CNNs are [batch, channel, height, width]
    def forward(self, x, skip):
        transposed = self.conv_transpose(x)
        cent_cropped = self.crop_center(skip, transposed.shape[2:])
        out = torch.cat([transposed, cent_cropped], 1) # We concatenate along dimension 1, since that refers to the channel dimension
        out = self.conv_block(out)

        return out


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(kwargs['input_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()
    

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)

                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    
    def forward(self, x_batch):
        batch_normalized = self.input_bn(x_batch)
        unet_output = self.unet(batch_normalized)
        final_pred_output = self.final(unet_output) # Will be probabilities
        
        return final_pred_output