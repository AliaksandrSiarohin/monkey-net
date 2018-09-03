from torch import nn
import torch.nn.functional as F
import torch
from modules.util import DownBlock3D, UpBlock3D, kp2gaussian, make_coordinate_grid


class PredictedDeformation(nn.Module):
    """
    Deformation module receive first frame and keypoints difference heatmap. It has hourglass architecture.
    """
    def __init__(self, block_expansion, num_kp, num_channels, kp_gaussian_sigma, relative=False):
        super(PredictedDeformation, self).__init__()

        out_channels_first = num_kp * max(1, block_expansion // 4)
        self.single_kp_conv = nn.Conv2d(3 * num_kp, out_channels_first, kernel_size=1, groups=num_kp)

        self.down_block1 = DownBlock3D(out_channels_first, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.down_block2 = DownBlock3D(block_expansion, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.down_block3 = DownBlock3D(block_expansion, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_block1 = UpBlock3D(block_expansion, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_block2 = UpBlock3D(2 * block_expansion, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_block3 = UpBlock3D(2 * block_expansion, block_expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv = nn.Conv3d(in_channels=block_expansion + out_channels_first, out_channels=2, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.kp_gaussian_sigma = kp_gaussian_sigma
        self.relative = relative

    def create_movement_encoding(self, kp_appearance, kp_video, spatial_size):
        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        kp_video = kp_video_diff + kp_appearance
        #kp_video_diff = torch.cat([kp_video[:, 0].unsqueeze(1), kp_video[:, :-1]], dim=1) - kp_video

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

    def predict(self, x):
        out1 = self.down_block1(x)
        out2 = self.down_block2(out1)
        out3 = self.down_block3(out2)

        out4 = self.up_block1(out3)
        out4 = torch.cat([out4, out2], dim=1)

        out5 = self.up_block2(out4)
        out5 = torch.cat([out5, out1], dim=1)

        out6 = self.up_block3(out5)
        out6 = torch.cat([out6, x], dim=1)

        out = self.conv(out6)

        return out

    def forward(self, appearance_frame, motion_video, kp_appearance, kp_video):
        kp_appearance = kp_appearance.detach()
        kp_video = kp_video.detach()

        bs, d, c, h, w = motion_video.shape

        movement_encoding = self.create_movement_encoding(kp_appearance, kp_video, (h, w))
        #appearance_encoding = appearance_frame.unsqueeze(2)
        #appearance_encoding = appearance_encoding.repeat(1, 1, movement_encoding.shape[2], 1, 1)

        #deformations_relative = self.predict(torch.cat([movement_encoding, appearance_encoding], dim=1))
        deformations_relative = self.predict(movement_encoding)
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        #deformations_relative = deformations_relative.cumsum(dim=-1)
        if self.relative:
            return deformations_relative

        coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)

        deformations_absolute = deformations_relative + coordinate_grid
        return deformations_absolute


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
        bs, d, c, h, w = motion_video.shape

        out = self.linear1(movement_encoding)
        out = F.relu(out)

        out = self.linear2(out)
        out = F.relu(out)

        out = self.linear3(out)
        out = out.view(-1, 2, 3)

        deformations_absolute = F.affine_grid(out, torch.Size((out.shape[0], 3, h, w)))

        return deformations_absolute


class IdentityDeformation(nn.Module):
    def forward(self, appearance_frame, motion_video, kp_appearance, kp_video):
        bs, d, c, h, w = motion_video.shape
        coordinate_grid = make_coordinate_grid((h, w), type=motion_video.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
        return coordinate_grid
