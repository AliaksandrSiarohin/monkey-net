from torch import nn
import torch
from modules.util import Decoder


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
    def __init__(self, block_expansion, num_channels, max_features, num_blocks,
                 embedding_features, use_kp_embedding):
        super(VideoDecoder, self).__init__()
        self.decoder = Decoder(block_expansion=block_expansion, in_features=num_channels,
                               out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                               dim=3, additional_features_for_block=embedding_features * int(use_kp_embedding))

    def forward(self, x):
        out = self.decoder(x)
        out = torch.sigmoid(out)
        return out
