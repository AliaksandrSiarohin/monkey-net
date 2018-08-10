import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import DeformationModule
from modules.video_decoder import VideoDecoder
from modules.util import KP2Gaussian
from modules.kp_decoder import KPDecoder

from modules.util import make_coordinate_grid


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, block_expansion, spatial_size, num_channels, num_kp, kp_gaussian_sigma):
        super(DDModel, self).__init__()

        self.appearance_encoder = AppearanceEncoder(block_expansion=block_expansion, num_channels=num_channels)
        self.deformation_module = DeformationModule(block_expansion=block_expansion,
                                                    num_channels=num_channels, num_kp=num_kp)
        self.kp2gaussian = KP2Gaussian(sigma=kp_gaussian_sigma, spatial_size=spatial_size)
        self.video_decoder = VideoDecoder(block_expansion=block_expansion, num_channels=num_channels)
        self.kp_decoder = KPDecoder(block_expansion=block_expansion, num_kp=num_kp)

        self.spatial_size = spatial_size

    def forward(self, appearance_frame, motion_video, kp_appearance=None, kp_video=None):
        appearance_skips = self.appearance_encoder(appearance_frame)

        bs, _, d, h, w = motion_video.shape

        if kp_video is None:
            # Extract keypoints
            kp_video = None

        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)

        if kp_appearance is None:
            # Extract keypoints
            kp_appearance = None

        kp_video = kp_video_diff + kp_appearance.unsqueeze(1)
        movement_encoding = self.kp2gaussian(kp_video)
        movement_encoding = movement_encoding.permute(0, 2, 1, 3, 4)
        appearance_encoding = appearance_frame.unsqueeze(2)
        appearance_encoding = appearance_encoding.repeat(1, 1, movement_encoding.shape[2], 1, 1)

        deformations_relative = self.deformation_module(torch.cat([movement_encoding, appearance_encoding], dim=1))
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        coordinate_grid = make_coordinate_grid(self.spatial_size, type=appearance_encoding.type())
        coordinate_grid = 2 * (coordinate_grid / self.spatial_size) - 1
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)

        deformations_absolute = deformations_relative + coordinate_grid
        deformations_absolute = deformations_absolute.view(-1, h, w, 2)
        deformations_absolute = deformations_absolute.permute(0, 3, 1, 2)

        deformed_skips = []
        for skip in appearance_skips:
            deformation_resize = F.interpolate(deformations_absolute, size=skip.shape[2:])
            deformation_resize = deformation_resize.permute(0, 2, 3, 1)

            deformed_skip = skip.unsqueeze(1).repeat(1, d, 1, 1, 1)

            shape = (-1, ) + skip.shape[1:]
            deformed_skip = deformed_skip.view(*shape)
            deformed_skip = F.grid_sample(deformed_skip, deformation_resize)

            shape = (bs, d) + skip.shape[1:]
            deformed_skip = deformed_skip.view(*shape)
            deformed_skip = deformed_skip.permute(0, 2, 1, 3, 4)

            deformed_skips.append(deformed_skip)

        out = self.video_decoder(deformed_skips)

        return out
