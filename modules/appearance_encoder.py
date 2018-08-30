from torch import nn
from modules.util import DownBlock2D


class AppearanceEncoder(nn.Module):
    """
    Encode appearance of the first video frame, return features from all blocks, for skip connections
    """
    def __init__(self, block_expansion, num_channels=3):
        super(AppearanceEncoder, self).__init__()

        self.conv =  nn.Conv2d(num_channels, block_expansion, kernel_size=(3, 3), padding=1)
        self.block1 = DownBlock2D(block_expansion, block_expansion)
        self.block2 = DownBlock2D(block_expansion, 2 * block_expansion)
        self.block3 = DownBlock2D(2 * block_expansion, 4 * block_expansion)
        self.block4 = DownBlock2D(4 * block_expansion, 8 * block_expansion)
        self.block5 = DownBlock2D(8 * block_expansion, 8 * block_expansion)

    def forward(self, x):
        out0 = self.conv(x)
        out1 = self.block1(out0)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        return [out0, out1, out2, out3, out4, out5]
