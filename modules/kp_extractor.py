from torch import nn
import torch.nn.functional as F
from modules.util import gaussian2kp, Hourglass


class KPExtractor(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, block_expansion, num_kp, num_channels, max_features, number_of_blocks, temperature=0.1):
        super(KPExtractor, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, number_of_blocks=number_of_blocks, dim=2)
        self.temperature = temperature

    def forward(self, x):
        out = self.predictor(x)

        out = gaussian2kp(out, self.temperature)

        return out
