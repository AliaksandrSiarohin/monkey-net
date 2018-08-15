from torch import nn
import torch.nn.functional as F
import torch
from modules.util import UpBlock3D


class WeightedSumBlock(nn.Module):
    def __init__(self, in_channels):
        super(WeightedSumBlock, self).__init__()

        self.weight_predict = nn.Conv3d(2 * in_channels, 1, kernel_size=1)

    def forward(self, x):
        weight = self.weight_predict(torch.cat(x, dim=1))
        weight = torch.sigmoid(weight)
        out = weight * x[0] + (1 - weight) * x[1]
        return out


class VideoDecoder(nn.Module):
    """
    Video decoder, take deformed feature maps and reconstruct a video.
    """
    def __init__(self, block_expansion, num_channels=3):
        super(VideoDecoder, self).__init__()

        self.block1 = UpBlock3D(8 * block_expansion, 8 * block_expansion)

        self.ws1 = WeightedSumBlock(8 * block_expansion)
        self.block2 = UpBlock3D(8 * block_expansion, 4 * block_expansion)

        self.ws2 = WeightedSumBlock(4 * block_expansion)
        self.block3 = UpBlock3D(4 * block_expansion, 2 * block_expansion)

        self.ws3 = WeightedSumBlock(2 * block_expansion)
        self.block4 = UpBlock3D(2 * block_expansion, block_expansion)

        self.ws4 = WeightedSumBlock(block_expansion)
        self.block5 = UpBlock3D(block_expansion, num_channels)

        self.ws5 = WeightedSumBlock(num_channels)
        #self.conv = nn.Conv3d(in_channels=num_channels,
        #                      out_channels=num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skip0, skip1, skip2, skip3, skip4, skip5 = x

        out = self.block1(skip5)

        out = self.ws1([out, skip4])
        out = self.block2(out)

        out = self.ws2([out, skip3])
        out = self.block3(out)

        out = self.ws3([out, skip2])
        out = self.block4(out)

        out = self.ws4([out, skip1])
        out = self.block5(out)

        out = torch.sigmoid(out)
        out = self.ws5([out, skip0])

        #out = self.conv(out)


        return out
