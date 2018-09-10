import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import PredictedDeformation, AffineDeformation, IdentityDeformation
from modules.video_decoder import VideoDecoder
from modules.kp_extractor import KPExtractor

from modules.util import kp2gaussian


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, kp_extractor_module_params, deformation_module_params, main_module_params, num_channels=3,
                 deformation_type='predicted', use_kp_embedding=True, kp_gaussian_sigma=2, num_kp=10):
        super(DDModel, self).__init__()

        assert deformation_type in ['affine', 'predicted', 'none']

        self.appearance_encoder = AppearanceEncoder(num_channels=num_channels, **main_module_params)

        if deformation_type == 'affine':
            self.deformation_module = AffineDeformation(num_kp=num_kp, **deformation_module_params)
        elif deformation_type == 'predicted':
            self.deformation_module = PredictedDeformation(kp_gaussian_sigma = kp_gaussian_sigma,
                                                           num_kp=num_kp, **deformation_module_params)
        else:
            self.deformation_module = IdentityDeformation()

        self.video_decoder = VideoDecoder(num_channels=num_channels, num_kp=num_kp, use_kp_embedding=use_kp_embedding,
                                          **main_module_params)
        self.kp_extractor = KPExtractor(num_channels=num_channels, num_kp=num_kp, **kp_extractor_module_params)
        self.use_kp_embedding = use_kp_embedding
        self.kp_gaussian_sigma = kp_gaussian_sigma

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w))
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def extract_kp(self, frames):
        kp = self.kp_extractor(frames)
        return kp

    def make_kp_embedding(self, kp_video, kp_appearance, appearance_skips):
        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        kp_shifted = kp_video_diff + kp_appearance
        bs, d, _, _ = kp_video.shape

        kp_skips = []
        for skip in appearance_skips:
            kp_emb = kp2gaussian(kp_shifted, spatial_size=skip.shape[3:], sigma=self.kp_gaussian_sigma)
            kp_emb = kp_emb.permute(0, 2, 1, 3, 4)
            kp_skips.append(kp_emb)

        return kp_skips

    def predict(self, appearance_frame, motion_video, kp_appearance, kp_video):
        appearance_skips = self.appearance_encoder(appearance_frame)

        deformations_absolute = self.deformation_module(appearance_frame, motion_video, kp_appearance, kp_video)
        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.use_kp_embedding:
            kp_skips = self.make_kp_embedding(kp_video, kp_appearance, appearance_skips)
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(appearance_frame, deformations_absolute)
        video_prediction = self.video_decoder(skips)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed,
                "deformation": deformations_absolute, "kp_array": kp_video}

    def forward(self, inp):
        return self.reconstruction(inp)

    def reconstruction(self, inp):
        motion_video = inp['video_array']
        appearance_frame = inp['video_array'][:, :, :1, :, :]

        if 'kp_array' in inp:
            kp_video = inp['kp_array']
            kp_appearance = inp['kp_array'][:, 0, :, :].unsqueeze(1)
        else:
            kp_video = self.extract_kp(motion_video)
            kp_appearance = kp_video[:, :1, :, :]

        return self.predict(appearance_frame, motion_video, kp_appearance, kp_video)

    def transfer(self, inp):
        motion_video = inp['first_video_array']
        appearance_frame = inp['second_video_array'][:, :, :1, :, :]

        if 'first_kp_array' in inp:
            kp_video = inp['first_kp_array']
            kp_appearance = inp['second_kp_array'][:, 0, :, :].unsqueeze(1)
        else:
            kp_video = self.extract_kp(motion_video)
            kp_appearance = self.extract_kp(appearance_frame)

        return self.predict(appearance_frame, motion_video, kp_appearance, kp_video)
