import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import PredictedDeformation, AffineDeformation
from modules.video_decoder import VideoDecoder
from modules.kp_decoder import KPDecoder

from modules.util import make_coordinate_grid

from skimage.transform import estimate_transform
import numpy as np


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, block_expansion, spatial_size, num_channels, num_kp, kp_gaussian_sigma, deformation_type):
        super(DDModel, self).__init__()

        assert deformation_type in ['affine', 'predicted']

        self.appearance_encoder = AppearanceEncoder(block_expansion=block_expansion, num_channels=num_channels)

        if deformation_type == 'affine':
            self.deformation_module = AffineDeformation(block_expansion=block_expansion, num_kp=num_kp,
                                                        spatial_size=spatial_size)
        else:
            self.deformation_module = PredictedDeformation(block_expansion=block_expansion, spatial_size=spatial_size,
                                                        kp_gaussian_sigma=kp_gaussian_sigma, num_channels=num_channels,
                                                        num_kp=num_kp)

        self.video_decoder = VideoDecoder(block_expansion=block_expansion, num_channels=num_channels)
        self.kp_decoder = KPDecoder(block_expansion=block_expansion, num_kp=num_kp)

        self.spatial_size = spatial_size

    def _deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, ch, h, w = inp.shape

        deformations_absolute = deformations_absolute.view(-1, h_old, w_old, 2)
        deformations_absolute = deformations_absolute.permute(0, 3, 1, 2)

        deformation = F.interpolate(deformations_absolute, size=(h, w))
        deformation = deformation.permute(0, 2, 3, 1)

        deformed_inp = inp.unsqueeze(1).repeat(1, d, 1, 1, 1)

        deformed_inp = deformed_inp.view(-1, ch, h, w)

        deformed_inp = F.grid_sample(deformed_inp, deformation)

        deformed_inp = deformed_inp.view(bs, d, ch, h, w)
        deformed_inp = deformed_inp.permute(0, 2, 1, 3, 4)

        return deformed_inp

    def forward(self, appearance_frame, motion_video, kp_appearance=None, kp_video=None):
        appearance_skips = self.appearance_encoder(appearance_frame)

        bs, _, d, h, w = motion_video.shape

        if kp_video is None:
            # Extract keypoints
            kp_video = None

        if kp_appearance is None:
            # Extract keypoints
            kp_appearance = None

        # kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        # kp_video = kp_video_diff + kp_appearance.unsqueeze(1)
        # movement_encoding = self.kp2gaussian(kp_video)
        # appearance_kp_encoding = self.kp2gaussian(kp_appearance)
        #
        # movement_encoding = movement_encoding - appearance_kp_encoding.unsqueeze(1)
        #
        # movement_encoding = movement_encoding.permute(0, 2, 1, 3, 4)
        #
        # appearance_encoding = appearance_frame.unsqueeze(2)
        # appearance_encoding = appearance_encoding.repeat(1, 1, movement_encoding.shape[2], 1, 1)
        #
        # deformations_relative = self.deformation_module(torch.cat([movement_encoding, appearance_encoding], dim=1))
        # deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)
        #
        # coordinate_grid = make_coordinate_grid(self.spatial_size, type=appearance_encoding.type())
        # coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
        #
        # deformations_absolute = deformations_relative + coordinate_grid

        # np_video_kp = kp_video.cpu().numpy()
        #
        # transforms = []
        # for b in range(np_video_kp.shape[0]):
        #     for dep in range(np_video_kp.shape[1]):
        #         transforms.append(estimate_transform('affine', np_video_kp[b, dep, :, ::-1], np_video_kp[b, 0, :, ::-1]).params[:2])
        #
        # transforms = np.array(transforms).reshape((bs * d, 2, 3)).astype('float32')
        #
        # #print (transforms)
        # transforms = torch.from_numpy(transforms).cuda(0)
        # deformations_absolute = F.affine_grid(transforms, torch.Size((bs * d, 3, h, w)))


        deformations_absolute = self.deformation_module(appearance_frame, motion_video, kp_appearance, kp_video)
        deformations_absolute = deformations_absolute.view(bs, d, h, w, 2)
        # index = torch.tensor([1, 0]).type(deformations_absolute.type()).long()
        # deformations_absolute = torch.index_select(deformations_absolute, -1, index)

        deformed_skips = [self._deform_input(skip, deformations_absolute) for skip in appearance_skips]
        video_deformed = self._deform_input(appearance_frame, deformations_absolute)

        video_prediction = self.video_decoder(deformed_skips)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed,
                "deformation": deformations_absolute, "kp_array": kp_video}
