from torch import nn
import torch
import torch.nn.functional as F


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3D, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.pad = nn.ReplicationPad3d((ka, kb, ka, kb, ka, kb))
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if out.shape[2] != 1:
            out = F.avg_pool3d(out, (2, 2, 2))
        else:
            out = F.avg_pool3d(out, (1, 2, 2))
        return out


class Discriminator(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, block_expansion=64, num_channels=3, num_blocks=4, max_features=512, kernel_size=4):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           norm=(i != 0),
                                           kernel_size=kernel_size))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        out_maps = [x]
        for down_block in self.down_blocks:
            out_maps.append(down_block(out_maps[-1]))
        score = self.conv(out_maps[-1])
        out_maps.append(score)
        return out_maps
