from torch import nn
import torch
import torch.nn.functional as F
from modules.util import make_coordinate_grid, matrix_inverse
from modules.kp_extractor import kp2gaussian


class MovementEmbeddingModule(nn.Module):
    """
    Produce embedding for movement
    """
    def __init__(self, num_kp, kp_variance, num_channels, use_deformed_appearance=False, use_difference=False,
                 use_heatmap=True, difference_type='absolute'):
        super(MovementEmbeddingModule, self).__init__()

        assert difference_type in ['absolute', 'relative']

        assert ((int(use_heatmap) + int(use_deformed_appearance) + int(use_difference)) >= 1)

        self.out_channels = (1 * use_heatmap + 2 * use_difference + num_channels * use_deformed_appearance) * num_kp

        self.kp_variance = kp_variance
        self.difference_type = difference_type
        self.use_difference = use_difference
        self.use_deformed_appearance = use_deformed_appearance
        self.use_heatmap = use_heatmap

    def forward(self, appearance_frame, kp_video, kp_appearance):
        spatial_size = appearance_frame.shape[3:]

        bs, _, _, h, w = appearance_frame.shape
        _, d, num_kp, _ = kp_video['mean'].shape

        inputs = []
        if self.use_heatmap:
            heatmap = kp2gaussian(kp_video, spatial_size=spatial_size, kp_variance=self.kp_variance)
            heatmap = heatmap.unsqueeze(3)
            inputs.append(heatmap)

        if self.use_difference or self.use_deformed_appearance:
            kp_video_diff = kp_appearance['mean'] - kp_video['mean']
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)

        if self.use_difference:
            inputs.append(kp_video_diff)

        if self.use_deformed_appearance:
            appearance_repeat = appearance_frame.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
            appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)

            deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1)
            coordinate_grid = make_coordinate_grid((h, w), type=deformation_approx.type())
            coordinate_grid = coordinate_grid.view(1, h, w, 2)
            deformation_approx = coordinate_grid + deformation_approx

            appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx)
            appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w))
            inputs.append(appearance_approx_deform)

        movement_encoding = torch.cat(inputs, dim=3)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)

        return movement_encoding.permute(0, 2, 1, 3, 4)
