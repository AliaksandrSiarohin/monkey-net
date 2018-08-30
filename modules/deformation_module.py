from torch import nn
import torch.nn.functional as F
import torch
from modules.util import ResBlock3D, kp2gaussian, make_coordinate_grid2d, add_z_coordinate


class PredictedDeformation(nn.Module):
    """
    Deformation module receive first frame and keypoints difference heatmap. It has hourglass architecture.
    """
    def __init__(self, block_expansion, num_kp, num_channels, kp_gaussian_sigma, num_blocks=1):
        super(PredictedDeformation, self).__init__()

        out_channels_first = num_kp * max(1, block_expansion // 4)
        self.single_kp_conv = nn.Conv2d(3 * num_kp, out_channels_first, kernel_size=1, groups=num_kp)

        blocks = []
        for i in range(num_blocks):
            blocks.append(ResBlock3D(out_channels_first + num_channels, merge='cat'))

        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv3d(in_channels=2 * (out_channels_first + num_channels), out_channels=2, kernel_size=3, padding=1)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.kp_gaussian_sigma = kp_gaussian_sigma

    def create_movement_encoding(self, kp_video, spatial_size):
        kp_video_diff = torch.cat([kp_video[:, 0].unsqueeze(1), kp_video[:, :-1]], dim=1) - kp_video

        movement_encoding = kp2gaussian(kp_video, spatial_size=spatial_size, sigma=self.kp_gaussian_sigma)

        bs, d, num_kp, h, w = movement_encoding.shape

        kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)
        movement_encoding = movement_encoding.unsqueeze(3)

        movement_encoding = torch.cat([movement_encoding, kp_video_diff], dim=3)
        movement_encoding = movement_encoding.view(bs * d, -1, h, w)
        movement_encoding = self.single_kp_conv(movement_encoding)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)
        # movement_encoding = kp_video_diff.view(bs, d, -1, h, w)
        # movement_encoding = movement_encoding[:, :, :2]

        return movement_encoding.permute(0, 2, 1, 3, 4)

    def predict(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.conv(out)
        return out

    def forward(self, appearance, kp_video):
        kp_video = kp_video.detach()

        bs, c, d, h, w = appearance.shape

        movement_encoding = self.create_movement_encoding(kp_video, (h, w))

        deformations_relative = self.predict(torch.cat([movement_encoding, appearance], dim=1))
        # print (movement_encoding.shape)
        #deformations_relative = movement_encoding
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        relative_to_first = deformations_relative.cumsum(dim=1)

        coordinate_grid = make_coordinate_grid2d((h, w), type=appearance.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)

        deformations_absolute = deformations_relative + coordinate_grid
        deformations_absolute = add_z_coordinate(deformations_absolute, deformations_absolute.type(), to_first=False)

        absolute_to_first = relative_to_first + coordinate_grid
        absolute_to_first = add_z_coordinate(absolute_to_first, absolute_to_first.type(), to_first=True)

        return {'absolute_to_previous': deformations_absolute, 'absolute_to_first': absolute_to_first,
                'relative_to_previous': deformations_relative, 'relative_to_first': relative_to_first}


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
        deformations_absolute = add_z_coordinate(deformations_absolute, deformations_absolute.type())

        return deformations_absolute
