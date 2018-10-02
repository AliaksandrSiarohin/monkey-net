from torch import nn
import torch.nn.functional as F
import torch
from modules.movement_embedding import MovementEmbeddingModule
from modules.non_local import NONLocalBlock3D


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3D, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.pad = nn.ReplicationPad3d((ka, kb, ka, kb, ka, kb))
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=(kernel_size, kernel_size, kernel_size))
        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        out = x
        out = self.pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        out = F.avg_pool3d(out, (1, 2, 2))
        return out


# class DownBlock3D(nn.Module):
#     """
#     Simple block for processing video (encoder).
#     """
#     def __init__(self, in_features, out_features, norm=False, pool=False, kernel_size=(1, 3, 3), padding=(0, 1, 1)):
#         super(DownBlock3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
#         if norm:
#             self.norm = nn.InstanceNorm3d(out_features, affine=True)
#         else:
#             self.norm = None
#         if pool:
#             self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
#         else:
#             self.pool = None
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.norm:
#             out = self.norm(out)
#         out = F.leaky_relu(out, 0.2)
#         if self.pool:
#             out = self.pool(out)
#         return out


class Discriminator(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, num_channels=3, num_kp=10, kp_variance=0.01,
                 block_expansion=64, num_blocks=4, max_features=512, use_kp=False, non_local_index=None):
        super(Discriminator, self).__init__()

        if use_kp:
            self.kp_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels,
                                                       use_difference=False, use_deformed_appearance=False)
            embedding_channels = self.kp_embedding.out_channels
        else:
            self.kp_embedding = None
            embedding_channels = 0

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           norm=(i != 0),
                                           kernel_size=4))
            if i == non_local_index:
                down_blocks.append(NONLocalBlock3D(min(max_features, block_expansion * (2 ** (i + 1))), bn_layer=False))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)

    def forward(self, x, kp_video, kp_appearance):
        out_maps = [x]
        if self.kp_embedding:
            heatmap = self.kp_embedding(x, kp_video, kp_appearance)
            out = torch.cat([x, heatmap], dim=1)
        else:
            out = x
        for down_block in self.down_blocks:
            out_maps.append(down_block(out))
            out = out_maps[-1]
        out = self.conv(out)
        out_maps.append(out)
        return out_maps
