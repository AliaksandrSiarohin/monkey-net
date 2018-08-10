from torch import nn
import torch.nn.functional as F
import torch
from modules.util import DownBlock3D, UpBlock3D


class DeformationModule(nn.Module):
    """
    Deformation module receive first frame and keypoints difference. It has hourglass architecture.
    """
    def __init__(self, block_expansion, num_kp, num_channels):
        super(DeformationModule, self).__init__()

        self.down_block1 = DownBlock3D(num_kp + num_channels, block_expansion)
        self.down_block2 = DownBlock3D(block_expansion, block_expansion)
        self.down_block3 = DownBlock3D(block_expansion, block_expansion)

        self.up_block1 = UpBlock3D(block_expansion, block_expansion)
        self.up_block2 = UpBlock3D(2 * block_expansion, block_expansion)
        self.up_block3 = UpBlock3D(2 * block_expansion, block_expansion)

        self.conv = nn.Conv3d(in_channels=block_expansion, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):

        out1 = self.down_block1(x)
        out2 = self.down_block2(out1)
        out3 = self.down_block3(out2)

        out4 = self.up_block1(out3)
        out4 = torch.cat([out4, out2], dim=1)

        out5 = self.up_block2(out4)
        out5 = torch.cat([out5, out1], dim=1)

        out6 = self.up_block3(out5)

        out = self.conv(out6)
        out = torch.tanh(out)

        return out
