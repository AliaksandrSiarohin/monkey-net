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

    def predict(self, appearance_frame, kp_video, kp_appearance):
        appearance_skips = self.appearance_encoder(appearance_frame)

        spatial_size = appearance_frame.shape[3:]
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
            d = kp_video['mean'].shape[1]
            movement_embedding = self.main_embedding_module(kp_appearance=kp_appearance, kp_video=kp_video,
                                                            spatial_size=spatial_size)
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


    def transfer(self, inp):
        #Batch size  should be 1
        assert inp['first_video_array'].shape[0] == 1

        motion_video = inp['first_video_array']
        appearance_frame = inp['second_video_array'][:, :, :1, :, :]

        kp_video = self.kp_extractor(motion_video)
        kp_appearance = self.kp_extractor(appearance_frame)

        best_frame = self.select_best_frame(kp_video['mean'], kp_appearance['mean'])

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
