from torch import nn
import torch.nn.functional as F
import torch
from modules.util import UpBlock3D, kp2gaussian
from modules.deformation_module import AffineDeformation, PredictedDeformation
from modules.losses import resize_kp


class WeightedSumBlock(nn.Module):
    def __init__(self, in_channels):
        super(WeightedSumBlock, self).__init__()
        self.weight_predict = nn.Conv3d(2 * in_channels, 1, kernel_size=1)

    def forward(self, x):
        weight = self.weight_predict(torch.cat(x, dim=1))
        weight = torch.sigmoid(weight)
        out = weight * x[0] + (1 - weight) * x[1]
        return out


class VideoDecoder(nn.Module):
    """
    Video decoder, take deformed feature maps and reconstruct a video.
    """
    def __init__(self, block_expansion, num_channels=3, num_kp=10, use_kp_encodings=True, use_deformation=True,
                 deformation_type='predicted', kp_gaussian_sigma=0.1, num_input_features=(8, 8, 4, 2, 1, 1)):
        super(VideoDecoder, self).__init__()

        assert deformation_type in ['affine', 'predicted']

        conv_modules = []
        deformation_modules = []
        merge_modules = []

        for i in range(len(num_input_features)):
            in_channels = num_input_features[i] * block_expansion
            if i != len(num_input_features) - 1:
                out_channels = num_input_features[i + 1] * block_expansion
                conv_modules.append(UpBlock3D(in_channels + use_kp_encodings * num_kp, out_channels))
            else:
                out_channels = num_channels
                conv_modules.append(nn.Conv3d(in_channels + use_kp_encodings * num_kp, out_channels, kernel_size=3, padding=1))

            if use_deformation:
                if deformation_type == 'affine':
                    deformation_modules.append(AffineDeformation(block_expansion=block_expansion, num_kp=num_kp))
                else:
                    deformation_modules.append(PredictedDeformation(block_expansion=block_expansion, kp_gaussian_sigma=kp_gaussian_sigma,
                                                                    num_channels=in_channels, num_kp=num_kp))

            merge_modules.append(WeightedSumBlock(in_channels))

        self.conv_modules = nn.ModuleList(conv_modules)
        if use_deformation:
            self.deformation_modules = nn.ModuleList(deformation_modules)

        self.merge_modules = nn.ModuleList(merge_modules)
        self.kp_gaussian_sigma = kp_gaussian_sigma
        self.use_deformation = use_deformation
        self.use_kp_encodings = use_kp_encodings

    def forward(self, skips, kp_video):
        out_dict = {'absolute_to_previous': [],
                    'absolute_to_first': [],
                    'relative_to_first': []}

        d = kp_video.shape[1] // 2 ** (len(self.conv_modules) - 1)

        out = skips[-1].unsqueeze(2).repeat(1, 1, d, 1, 1)

        for i, skip in enumerate(skips[::-1]):
            bs, c, h, w = skip.shape
            kp_video_resized = resize_kp(kp_video, d / kp_video.shape[1])

            if self.use_deformation:
                deformations = self.deformation_modules[i](out, kp_video_resized)
                out_dict['absolute_to_previous'].append(deformations['absolute_to_previous'])
                out_dict['relative_to_first'].append(deformations['relative_to_first'])
                out_dict['absolute_to_first'].append(deformations['absolute_to_first'])
                skip_deformed = F.grid_sample(skip.unsqueeze(2), grid=deformations['absolute_to_first'].detach())
            else:
                skip_deformed = skip.unsqueeze(2).repeat(1, 1, d, 1, 1)
            out = self.merge_modules[i]([out, skip_deformed])

            if self.use_kp_encodings:
                kp_encoding = kp2gaussian(kp_video_resized, spatial_size=(h, w), sigma=self.kp_gaussian_sigma)
                kp_encoding = kp_encoding.permute(0, 2, 1, 3, 4)
                out = self.conv_modules[i](torch.cat([out, kp_encoding], dim=1))
            else:
                out = self.conv_modules[i](out)

            d *= 2
        out_dict['video_prediction'] = torch.sigmoid(out)

        return out_dict
