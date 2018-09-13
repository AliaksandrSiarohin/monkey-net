from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, Encoder


class PredictedDeformation(nn.Module):
    """
    Deformation module receive first frame and keypoints difference heatmap. It has hourglass architecture.
    """
    def __init__(self, block_expansion, number_of_blocks, max_features, embedding_features, cumsum=False, relative=False):
        super(PredictedDeformation, self).__init__()
        self.predictor = Hourglass(block_expansion=block_expansion, in_features=embedding_features, out_features=2,
                                   max_features=max_features, dim=3, number_of_blocks=number_of_blocks)
        self.predictor.decoder.conv.weight.data.zero_()
        self.predictor.decoder.conv.bias.data.zero_()

        self.cumsum = cumsum
        self.relative = relative

    def forward(self, movement_embedding):
        h, w = movement_embedding.shape[3:]

        deformations_relative = self.predictor(movement_embedding)
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        if self.cumsum:
            deformations_relative = deformations_relative.cumsum(dim=-1)

        if self.relative:
            deformation = deformations_relative
        else:
            coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
            coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
            deformation = deformations_relative + coordinate_grid
        z_coordinate = torch.zeros(deformation.shape[:-1] + (1, )).type(deformation.type())

        return torch.cat([deformation, z_coordinate], dim=-1)


class AffineDeformation(nn.Module):
    """
    Deformation module receive keypoint cordinates and try to predict values for affine deformation. It has MPL architecture.
    """
    def __init__(self, block_expansion, number_of_blocks, max_features, embedding_features, cumsum):
        super(AffineDeformation, self).__init__()

        self.encoder = Encoder(block_expansion, in_features=embedding_features, max_features=max_features,
                               dim=2, number_of_blocks=number_of_blocks)

        self.linear = nn.Linear(in_features=self.encoder.down_blocks[-1].out_features, out_features=6)

        self.linear.weight.data.zero_()
        self.linear.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.cumsum = cumsum

    def forward(self, movement_embedding):
        bs, c, d, h, w = movement_embedding.shape[3:]

        out = self.encoder(movement_embedding)
        out = out.permute(0, 2, 1, 3, 4).view(bs * d, c, h, w)
        out = out.mean(dim=(2, 3))
        out = self.linear(out)

        deformations_absolute = F.affine_grid(out, torch.Size((out.shape[0], 3, h, w)))

        deformations_absolute = deformations_absolute.view(bs, d, h, w, 2)

        if self.cumsum:
            coordinate_grid = make_coordinate_grid((h, w), type=deformations_absolute.type())
            coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
            deformation_relative = deformations_absolute - coordinate_grid
            deformation_relative = deformation_relative.cumsum(dim=-1)
            deformations_absolute = deformation_relative + coordinate_grid

        z_coordinate = torch.zeros(deformations_absolute.shape[:-1] + (1, )).type(deformations_absolute.type())
        return torch.cat([deformations_absolute, z_coordinate], dim=-1)


class IdentityDeformation(nn.Module):
    def forward(self, movement_embedding):
        bs, c, d, h, w = movement_embedding.shape
        coordinate_grid = make_coordinate_grid((h, w), type=movement_embedding.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)

        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1, )).type(coordinate_grid.type())
        return torch.cat([coordinate_grid, z_coordinate], dim=-1)
