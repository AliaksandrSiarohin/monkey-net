from torch import nn
import torch.nn.functional as F
import torch
from modules.util import UpBlock3D


class VideoDecoder(nn.Module):
    """
    Video decoder, take deformed feature maps and reconstruct a video.
    """
    def __init__(self, block_expansion, num_channels=3):
        super(VideoDecoder, self).__init__()

        self.block1 = UpBlock3D(8 * block_expansion, 8 * block_expansion)
        self.block2 = UpBlock3D(16 * block_expansion, 4 * block_expansion)
        self.block3 = UpBlock3D(8 * block_expansion, 2 * block_expansion)
        self.block4 = UpBlock3D(4 * block_expansion, block_expansion)
        self.block5 = UpBlock3D(2 * block_expansion, block_expansion)

        self.conv = nn.Conv3d(in_channels=block_expansion, out_channels=num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skip1, skip2, skip3, skip4, skip5 = x

        out = self.block1(skip5)

        out = torch.cat([out, skip4], dim=1)
        out = self.block2(out)

        out = torch.cat([out, skip3], dim=1)
        out = self.block3(out)

        out = torch.cat([out, skip2], dim=1)
        out = self.block4(out)

        out = torch.cat([out, skip1], dim=1)
        out = self.block5(out)

        out = self.conv(out)

        final_shape = out.shape
        out = out.view(final_shape[0], final_shape[1], -1)
        out = F.softmax(out, dim=2)
        out = out.view(*final_shape)

        return out
