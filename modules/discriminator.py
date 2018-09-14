from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Encoder


class Discriminator(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, block_expansion, num_channels, max_features, number_of_blocks):
        super(Discriminator, self).__init__()

        self.predictor = Encoder(block_expansion, in_features=num_channels,
                                 max_features=max_features, number_of_blocks=number_of_blocks, dim=3)

    def forward(self, x):
        out_maps = self.predictor(x)
        return out_maps
