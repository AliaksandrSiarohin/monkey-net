import torch
from torch import nn
import torch.nn.functional as F

from modules.appearance_encoder import AppearanceEncoder
from modules.video_decoder import VideoDecoder
from modules.kp_extractor import KPExtractor


class DDModel(nn.Module):
    """
    Deformable disentangling model for videos.
    """
    def __init__(self, block_expansion, num_channels, num_kp, kp_gaussian_sigma, deformation_type,
                 kp_extractor_temperature):
        super(DDModel, self).__init__()

        self.appearance_encoder = AppearanceEncoder(block_expansion=block_expansion, num_channels=num_channels)
        self.video_decoder = VideoDecoder(block_expansion=block_expansion, num_channels=num_channels, num_kp=num_kp,
                                          deformation_type=deformation_type, kp_gaussian_sigma=kp_gaussian_sigma)
        self.kp_extractor = KPExtractor(block_expansion=block_expansion, num_kp=num_kp, num_channels=num_channels,
                                        temperature=kp_extractor_temperature)

        self.kp_gaussian_sigma = kp_gaussian_sigma

    def extract_kp(self, frames):
        bs, c, d, h, w = frames.shape
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().view(bs * d, c, h, w)
        kp = self.kp_extractor(frames)
        _, num_kp, _ = kp.shape
        return kp.view(bs, d, num_kp, 2)

    def predict(self, appearance_frame, motion_video, kp_appearance, kp_video):
        appearance_skips = self.appearance_encoder(appearance_frame)
        bs, _, d, h, w = motion_video.shape

        kp_video_diff = kp_video - kp_video[:, 0].unsqueeze(1)
        kp_shifted = kp_video_diff + kp_appearance

        prediction_dict = self.video_decoder(appearance_skips, kp_shifted)
        prediction_dict['kp_array'] = kp_video

        if len(prediction_dict['absolute_to_first']) != 0:
            prediction_dict['video_deformed'] = F.grid_sample(appearance_frame.unsqueeze(2), prediction_dict['absolute_to_first'][-1])
        else:
            prediction_dict['video_deformed'] = prediction_dict['video_prediction']

        return prediction_dict

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
