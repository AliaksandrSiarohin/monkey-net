import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import PredictedDeformation, AffineDeformation
from modules.video_decoder import VideoDecoder
from modules.kp_extractor import KPExtractor

from modules.util import make_coordinate_grid
import numpy as np

class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, block_expansion, spatial_size, num_channels, num_kp, kp_gaussian_sigma, deformation_type,
                 kp_extractor_temperature):
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
        self.kp_extractor = KPExtractor(block_expansion=block_expansion, num_kp=num_kp, num_channels=num_channels,
                                        temperature=kp_extractor_temperature)

        self.spatial_size = spatial_size

    def deform_input(self, inp, deformations_absolute):
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

    def extract_kp(self, frames):
        bs, c, d, h, w = frames.shape
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().view(bs * d, c, h, w)
        kp = self.kp_extractor(frames)
        _, num_kp, _ = kp.shape
        return kp.view(bs, d, num_kp, 2)

    def predict(self, appearance_frame, motion_video, kp_appearance, kp_video):
        appearance_skips = self.appearance_encoder(appearance_frame)

        bs, _, d, h, w = motion_video.shape

        deformations_absolute = self.deformation_module(appearance_frame, motion_video, kp_appearance, kp_video)
        deformations_absolute = deformations_absolute.view(bs, d, h, w, 2)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]
        video_deformed = self.deform_input(appearance_frame, deformations_absolute)

        video_prediction = self.video_decoder(deformed_skips)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed,
                "deformation": deformations_absolute, "kp_array": kp_video}

    def forward(self, inp, transfer=False):
        if transfer:
            return self.transfer(inp)
        else:
            return self.reconstruction(inp)

    def reconstruction(self, inp):
        motion_video = inp['video_array']
        appearance_frame = inp['video_array'][:, :, 0, :, :]

        if 'kp_array' in inp:
            kp_video = inp['kp_array']
            kp_appearance = inp['kp_array'][:, 0, :, :].unsqueeze(1)
        else:
            kp_video = self.extract_kp(motion_video)
            kp_appearance = kp_video[:, 0, :, :].unsqueeze(1)

        return self.predict(appearance_frame, motion_video, kp_appearance, kp_video)

    def transfer(self, inp):
        motion_video = inp['first_video_array']

        appearance_frame = inp['second_video_array'][:, :, 0, :, :]

        if 'first_kp_array' in inp:
            kp_video = inp['first_kp_array']
            kp_appearance = inp['second_kp_array'][:, 0, :, :].unsqueeze(1)
        else:
            kp_video = self.extract_kp(motion_video)
            kp_appearance = self.extract_kp(appearance_frame.unsqueeze(2))

        return self.predict(appearance_frame, motion_video, kp_appearance, kp_video)
