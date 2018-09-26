import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import DeformationModule, IdentityDeformation
from modules.video_decoder import VideoDecoder
from modules.movement_embedding import MovementEmbeddingModule


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, num_channels, num_kp, kp_variance,
                 main_module_params,
                 deformation_module_params=None, kp_embedding_params=None, detach_deformation=False):
        super(DDModel, self).__init__()

        self.appearance_encoder = AppearanceEncoder(num_channels=num_channels, **main_module_params)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
        else:
            self.kp_embedding_module = None

        if deformation_module_params is not None:
            self.deformation_module = DeformationModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels,
                                                        **deformation_module_params)
        else:
            self.deformation_module = IdentityDeformation()

        self.video_decoder = VideoDecoder(num_channels=num_channels, embedding_features=self.kp_embedding_module.out_channels,
                                          use_kp_embedding=kp_embedding_params is not None, **main_module_params)

        self.detach_deformation = detach_deformation

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w))
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def forward(self, appearance_frame, kp_video):
        appearance_skips = self.appearance_encoder(appearance_frame)

        if self.detach_deformation:
            deformations_absolute = self.deformation_module(kp_video={k:v.detach() for k, v in kp_video.items()},
                                                            appearance_frame=appearance_frame)
        else:
            deformations_absolute = self.deformation_module(kp_video=kp_video, appearance_frame=appearance_frame)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_video['mean'].shape[1]
            movement_embedding = self.kp_embedding_module(kp_video=kp_video, appearance_frame=appearance_frame)
            kp_skips = [F.interpolate(movement_embedding, size=(d, ) + skip.shape[3:]) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(appearance_frame, deformations_absolute)
        video_prediction = self.video_decoder(skips)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed,
                'deformation': deformations_absolute}
