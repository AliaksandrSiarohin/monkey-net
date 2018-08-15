from torch import nn
import torch.nn.functional as F
import torch
from modules.util import UpBlock2D, DownBlock2D, make_coordinate_grid


class KPExtractor(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, block_expansion, num_kp, num_channels, temperature=0.1):
        super(KPExtractor, self).__init__()

        self.down_block1 = DownBlock2D(num_channels, block_expansion)
        self.down_block2 = DownBlock2D(block_expansion, 2 * block_expansion)
        self.down_block3 = DownBlock2D(2 * block_expansion, 4 * block_expansion)
        self.down_block4 = DownBlock2D(4 * block_expansion, 8 * block_expansion)
        self.down_block5 = DownBlock2D(8 * block_expansion, 8 * block_expansion)

        self.up_block1 = UpBlock2D(8 * block_expansion, 8 * block_expansion)
        self.up_block2 = UpBlock2D(16 * block_expansion, 4 * block_expansion)
        self.up_block3 = UpBlock2D(8 * block_expansion, 2 * block_expansion)
        self.up_block4 = UpBlock2D(4 * block_expansion, block_expansion)
        self.up_block5 = UpBlock2D(2 * block_expansion, block_expansion)

        self.conv = nn.Conv2d(in_channels=block_expansion + num_channels, out_channels=num_kp, kernel_size=3, padding=1)

        self.temperature = temperature

    def forward(self, x):

        out1 = self.down_block1(x)
        out2 = self.down_block2(out1)
        out3 = self.down_block3(out2)
        out4 = self.down_block4(out3)
        out5 = self.down_block5(out4)

        out = self.up_block1(out5)

        out = torch.cat([out, out4], dim=1)
        out = self.up_block2(out)

        out = torch.cat([out, out3], dim=1)
        out = self.up_block3(out)

        out = torch.cat([out, out2], dim=1)
        out = self.up_block4(out)

        out = torch.cat([out, out1], dim=1)
        out = self.up_block5(out)

        out = torch.cat([out, x], dim=1)
        out = self.conv(out)

        final_shape = out.shape
        out = out.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(out / self.temperature, dim=2)
        out = heatmap.view(final_shape + (1, ))

        grid = make_coordinate_grid(final_shape[2], x.type()).unsqueeze(0).unsqueeze(1)

        out = (out * grid).sum(dim=(2, 3))

        return out
