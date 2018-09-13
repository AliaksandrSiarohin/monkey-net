import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import PredictedDeformation, AffineDeformation, IdentityDeformation
from modules.video_decoder import VideoDecoder
from modules.kp_extractor import KPExtractor, MovementEmbeddingModule


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, kp_extractor_module_params, deformation_module_params, main_module_params, deformation_embedding_params,
                 main_embedding_params, num_channels, detach_deformation, deformation_type, use_kp_embedding, kp_variance, num_kp):
        super(DDModel, self).__init__()

        assert deformation_type in ['affine', 'predicted', 'none']

        self.appearance_encoder = AppearanceEncoder(num_channels=num_channels, **main_module_params)
        self.deformation_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                                    **deformation_embedding_params)
        self.main_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                             **main_embedding_params)

        if deformation_type == 'affine':
            self.deformation_module = AffineDeformation(embedding_features=self.deformation_embedding_module.out_channels,
                                                        **deformation_module_params)
        elif deformation_type == 'predicted':
            self.deformation_module = PredictedDeformation(embedding_features=self.deformation_embedding_module.out_channels,
                                                           **deformation_module_params)
        else:
            self.deformation_module = IdentityDeformation()

        self.video_decoder = VideoDecoder(num_channels=num_channels, embedding_features=self.main_embedding_module.out_channels,
                                          use_kp_embedding=use_kp_embedding, **main_module_params)
        self.kp_extractor = KPExtractor(num_channels=num_channels, num_kp=num_kp, kp_variance=kp_variance,
                                        **kp_extractor_module_params)
        self.use_kp_embedding = use_kp_embedding
        self.detach_deformation = detach_deformation

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w))
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def predict(self, appearance_frame, motion_video):
        kp_video = self.kp_extractor(motion_video)
        kp_appearance = self.kp_extractor(appearance_frame)

        appearance_skips = self.appearance_encoder(appearance_frame)

        spatial_size = motion_video.shape[3:]
        if self.detach_deformation:
            movement_embedding = self.deformation_embedding_module(kp_appearance={k:v.detach() for k, v in kp_appearance.items()},
                                                                   kp_video={k:v.detach() for k, v in kp_video.items()},
                                                                   spatial_size=spatial_size)
        else:
            movement_embedding = self.deformation_embedding_module(kp_appearance=kp_appearance,
                                                                   kp_video=kp_video,
                                                                   spatial_size=spatial_size)

        deformations_absolute = self.deformation_module(movement_embedding)
        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.use_kp_embedding:
            d = motion_video.shape[2]
            movement_embedding = self.main_embedding_module(kp_appearance=kp_appearance, kp_video=kp_video,
                                                            spatial_size=spatial_size)
            kp_skips = [F.interpolate(movement_embedding, size=(d, ) + skip.shape[3:]) for skip in appearance_skips]
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
        return self.predict(appearance_frame, motion_video)

    def transfer(self, inp):
        motion_video = inp['first_video_array']
        appearance_frame = inp['second_video_array'][:, :, :1, :, :]
        return self.predict(appearance_frame, motion_video)
