from torch import nn
import torch.nn.functional as F
from modules.util import make_coordinate_grid, Hourglass


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

        final_shape = out.shape
        out = out.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(out / self.temperature, dim=3)
        out = heatmap.view(final_shape + (1, ))

        grid = make_coordinate_grid(final_shape[3:], x.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

        out = (out * grid).sum(dim=(3, 4))

        return out.permute(0, 2, 1, 3)
