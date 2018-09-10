from torch import nn
from modules.util import Encoder


class AppearanceEncoder(nn.Module):
    """
    Encode appearance of the first video frame, return features from all blocks, for skip connections
    """
    def __init__(self, block_expansion, num_channels=3, number_of_blocks=5, max_features=128):
        super(AppearanceEncoder, self).__init__()
        self.encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                               dim=2, number_of_blocks=number_of_blocks)

    def forward(self, x):
        return self.encoder(x)
