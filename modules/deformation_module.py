from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, kp2gaussian, make_coordinate_grid


class PredictedDeformation(nn.Module):
    """
    Deformation module receive first frame and keypoints difference heatmap. It has hourglass architecture.
    """
    def __init__(self, block_expansion, number_of_blocks, max_features, num_kp, kp_gaussian_sigma, relative=False):
        super(PredictedDeformation, self).__init__()

        out_channels_first = num_kp * max(1, block_expansion // 4)
        self.single_kp_conv = nn.Conv2d(3 * num_kp, out_channels_first, kernel_size=1, groups=num_kp)

        self.predictor = Hourglass(block_expansion=block_expansion, in_features=out_channels_first, out_features=2,
                                   max_features=max_features, dim=3, number_of_blocks=number_of_blocks)

        self.predictor.decoder.conv.weight.data.zero_()
        self.predictor.decoder.conv.bias.data.zero_()

        self.kp_gaussian_sigma = kp_gaussian_sigma
        self.relative = relative

    def create_movement_encoding(self, kp_appearance, kp_video, spatial_size):
        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        kp_video = kp_video_diff + kp_appearance

        movement_encoding = kp2gaussian(kp_video, spatial_size=spatial_size, sigma=self.kp_gaussian_sigma)

        movement_encoding = movement_encoding - movement_encoding[:, 0].unsqueeze(1)

        bs, d, num_kp, h, w = movement_encoding.shape

        kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)
        movement_encoding = movement_encoding.unsqueeze(3)

        movement_encoding = torch.cat([movement_encoding, kp_video_diff], dim=3)
        movement_encoding = movement_encoding.view(bs * d, -1, h, w)
        movement_encoding = self.single_kp_conv(movement_encoding)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)

        return movement_encoding.permute(0, 2, 1, 3, 4)

    def forward(self, appearance_frame, motion_video, kp_appearance, kp_video):
        kp_appearance = kp_appearance.detach()
        kp_video = kp_video.detach()

        _, _, _, h, w = motion_video.shape

        movement_encoding = self.create_movement_encoding(kp_appearance, kp_video, (h, w))

        deformations_relative = self.predictor(movement_encoding)
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        if self.relative:
            return deformations_relative

        coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)

        deformations_absolute = deformations_relative + coordinate_grid
        z_coordinate = torch.zeros(deformations_absolute.shape[:-1] + (1, )).type(deformations_absolute.type())
        return torch.cat([deformations_absolute, z_coordinate], dim=-1)


class AffineDeformation(nn.Module):
    """
    Deformation module receive keypoint cordinates and try to predict values for affine deformation. It has MPL architecture.
    """
    def __init__(self, block_expansion, num_kp):
        super(AffineDeformation, self).__init__()

        self.linear1 = nn.Linear(2 * 2 * num_kp, 4 * block_expansion)
        self.linear2 = nn.Linear(4 * block_expansion, 4 * block_expansion)
        self.linear3 = nn.Linear(4 * block_expansion, 6)

        self.linear3.weight.data.zero_()
        self.linear3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def create_movement_encoding(self, kp_appearance, kp_video):
        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        kp_video = kp_video_diff + kp_appearance

        kp_old = kp_video[:, 0].unsqueeze(1).repeat(1, kp_video.shape[1], 1, 1)

        out = torch.cat([kp_video, kp_old], dim=2)
        return out.view(kp_video.shape[0] * kp_video.shape[1], -1)

    def forward(self, appearance_frame, motion_video, kp_appearance, kp_video):
        movement_encoding = self.create_movement_encoding(kp_appearance, kp_video)
        bs, _, _, h, w = motion_video.shape

        out = self.linear1(movement_encoding)
        out = F.relu(out)

        out = self.linear2(out)
        out = F.relu(out)

        out = self.linear3(out)
        out = out.view(-1, 2, 3)

        deformations_absolute = F.affine_grid(out, torch.Size((out.shape[0], 3, h, w)))
        z_coordinate = torch.zeros(deformations_absolute.shape[:-1] + (1, )).type(deformations_absolute.type())
        return torch.cat([deformations_absolute, z_coordinate], dim=-1)


class IdentityDeformation(nn.Module):
    def forward(self, appearance_frame, motion_video, kp_appearance, kp_video):
        bs, c, d, h, w = motion_video.shape
        coordinate_grid = make_coordinate_grid((h, w), type=motion_video.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)

        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1, )).type(coordinate_grid.type())
        return torch.cat([coordinate_grid, z_coordinate], dim=-1)
