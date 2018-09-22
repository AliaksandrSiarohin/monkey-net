import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.deformation_module import DeformationModule, IdentityDeformation
from modules.video_decoder import VideoDecoder
from modules.movement_embedding import MovementEmbeddingModule
from modules.kp_extractor import KPExtractor


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, num_channels, num_kp, kp_variance,
                 kp_extractor_module_params, main_module_params,
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
        self.kp_extractor = KPExtractor(num_channels=num_channels, num_kp=num_kp, kp_variance=kp_variance,
                                        **kp_extractor_module_params)
        self.detach_deformation = detach_deformation

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w))
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def predict(self, appearance_frame, kp_video, kp_appearance):
        appearance_skips = self.appearance_encoder(appearance_frame)

        if self.detach_deformation:
            deformations_absolute = self.deformation_module(kp_appearance={k:v.detach() for k, v in kp_appearance.items()},
                                                                   kp_video={k:v.detach() for k, v in kp_video.items()},
                                                                   appearance_frame=appearance_frame)
        else:
            deformations_absolute = self.deformation_module(kp_appearance=kp_appearance,
                                                                   kp_video=kp_video,
                                                                   appearance_frame=appearance_frame)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_video['mean'].shape[1]
            movement_embedding = self.kp_embedding_module(kp_appearance=kp_appearance, kp_video=kp_video,
                                                            appearance_frame=appearance_frame)
            kp_skips = [F.interpolate(movement_embedding, size=(d, ) + skip.shape[3:]) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(appearance_frame, deformations_absolute)
        video_prediction = self.video_decoder(skips)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed,
                'deformation': deformations_absolute}

    def forward(self, inp):
        return self.reconstruction(inp)

    def reconstruction(self, inp):
        motion_video = inp['video_array']
        appearance_frame = inp['video_array'][:, :, :1, :, :]

        kp_video = self.kp_extractor(motion_video)
        kp_appearance = self.kp_extractor(appearance_frame)

        out = self.predict(appearance_frame, kp_video, kp_appearance)

        out['kp_video'] = kp_video

        return out

    def compute_pairwise_distances(self, kp_array):
        bs, d, num_kp, _ = kp_array.shape

        distances = torch.zeros(bs, d, num_kp, num_kp)

        for i in range(num_kp):
            for j in range(num_kp):
                distances[:, :, i, j] = torch.abs(kp_array[:, :, i] - kp_array[:, :, j]).sum(dim=-1)

        distances = distances.view(bs, d, -1)
        median = distances.median(dim=-1, keepdim=True)[0]
        distances /= median

        return distances

    def select_best_frame(self, kp_video, kp_appearance):
        video_distances = self.compute_pairwise_distances(kp_video)
        appearance_distances = self.compute_pairwise_distances(kp_appearance)

        norm = torch.abs(video_distances - appearance_distances).sum(dim=-1)

        best_frame = torch.argmin(norm, dim=-1)
        return best_frame.squeeze(dim=0)

    def transfer(self, inp, select_best_frame=False):
        #Batch size  should be 1
        assert inp['first_video_array'].shape[0] == 1

        motion_video = inp['first_video_array']
        appearance_frame = inp['second_video_array'][:, :, :1, :, :]

        kp_video = self.kp_extractor(motion_video)
        kp_appearance = self.kp_extractor(appearance_frame)

        if select_best_frame:
            best_frame = self.select_best_frame(kp_video['mean'], kp_appearance['mean'])
        else:
            best_frame = 0

        reverse_sample = range(0, best_frame + 1)[::-1]
        first_video_seq = {k: v[:, reverse_sample] for k, v in kp_video.items()}
        first_out = self.predict(appearance_frame, first_video_seq, kp_appearance)

        second_video_seq = {k: v[:, best_frame:] for k, v in kp_video.items()}
        second_out = self.predict(appearance_frame, second_video_seq, kp_appearance)

        out = dict()

        sample = range(1, best_frame + 1)[::-1]
        out['video_prediction'] = torch.cat([first_out['video_prediction'][:, :, sample],
                                             second_out['video_prediction']], dim=2)
        out['video_deformed'] = torch.cat([first_out['video_deformed'][:, :, sample],
                                           second_out['video_prediction']], dim=2)

        out['kp_video'] = kp_video
        out['kp_appearance'] = kp_appearance

        return out
