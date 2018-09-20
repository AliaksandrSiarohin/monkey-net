from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, matrix_inverse


def kp2gaussian(kp, spatial_size, kp_variance='learned'):
    mean = kp['mean']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1, ) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'learned':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    out_shape = out.shape
    out = out.view(out_shape[0], out_shape[1], out_shape[2], -1)
    heatmap = out / out.sum(dim=3, keepdim=True)
    out = heatmap.view(*out_shape)

    return out


def gaussian2kp(heatmap, kp_variance='learned'):
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

    mean = (heatmap * grid).sum(dim=(3, 4))

    kp = {'mean': mean.permute(0, 2, 1, 3)}

    if kp_variance == 'learned':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(3, 4))
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    return kp


class KPExtractor(nn.Module):
    """
    Extractor of keypoints. Return kp feature maps.
    """
    def __init__(self, block_expansion, num_kp, num_channels, max_features, number_of_blocks, temperature,
                 kp_variance):
        super(KPExtractor, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, number_of_blocks=number_of_blocks, dim=2)
        self.temperature = temperature
        self.kp_variance = kp_variance

    def forward(self, x):
        heatmap = self.predictor(x)

        final_shape = heatmap.shape
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=3)
        heatmap = heatmap.view(*final_shape)

        out = gaussian2kp(heatmap, self.kp_variance)

        return out


class MovementEmbeddingModule(nn.Module):
    """
    Produce embedding for movement
    """
    def __init__(self, num_kp, use_difference, kp_variance, num_channels,
                 use_deformed_appearance=False, learnable=False,
                 features_per_kp=1, difference_type='absolute', aggregation_type='cat'):
        super(MovementEmbeddingModule, self).__init__()

        assert difference_type in ['absolute', 'relative']
        assert aggregation_type in ['cat', 'sum']

        if learnable:
            self.out_channels = features_per_kp * num_kp
            self.conv = nn.Conv2d((1 + 2 * use_difference + num_channels * use_deformed_appearance) * num_kp,
                                  self.out_channels, kernel_size=1, groups=num_kp)
        else:
            self.out_channels = (1 + 2 * use_difference + num_channels * use_deformed_appearance) * num_kp

        if aggregation_type == 'sum':
            self.out_channels = self.out_channels // num_kp

        self.learnable = learnable
        self.kp_variance = kp_variance
        self.difference_type = difference_type
        self.use_difference = use_difference
        self.use_deformed_appearance = use_deformed_appearance
        self.aggregation_type = aggregation_type

    def combine_kp(self, kp_appearance, kp_video):
        kp_video_diff = kp_video['mean'] - kp_video['mean'][:, 0:1]
        kp_mean = kp_video_diff + kp_appearance['mean']
        out = {'mean': kp_mean}

        if self.kp_variance == 'learned':
            kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1]))
            kp_var = torch.matmul(kp_var, kp_appearance['var'])
            out['var'] = kp_var

        if self.difference_type == 'relative':
            kp_video_diff = torch.cat([kp_video['mean'][:, 0:1], kp_video['mean'][:, :-1]], dim=1) - kp_video['mean']
        else:
            kp_video_diff = kp_video_diff

        return out, kp_video_diff

    def forward(self, kp_appearance, kp_video, appearance_frame=None):
        spatial_size = appearance_frame.shape[3:]
        kp_video, kp_video_diff = self.combine_kp(kp_appearance, kp_video)

        movement_encoding = kp2gaussian(kp_video, spatial_size=spatial_size, kp_variance=self.kp_variance)

        if self.difference_type == 'relative':
            movement_encoding = torch.cat([movement_encoding[:, 0:1], movement_encoding[:, :-1]], dim=1) - movement_encoding
        else:
            movement_encoding = movement_encoding - movement_encoding[:, 0:1]
        bs, d, num_kp, h, w = movement_encoding.shape

        if self.use_difference or self.use_deformed_appearance:
            movement_encoding = movement_encoding.unsqueeze(3)
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)

            inputs = [movement_encoding]
            if self.use_difference:
                inputs.append(kp_video_diff)

            if self.use_deformed_appearance:
                appearance_repeat = appearance_frame.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
                appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)

                deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1)
                coordinate_grid = make_coordinate_grid((h, w), type=deformation_approx.type())
                coordinate_grid = coordinate_grid.view(1, h, w, 2)
                deformation_approx = coordinate_grid - deformation_approx

                appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx)
                appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w))
                inputs.append(appearance_approx_deform)

            movement_encoding = torch.cat(inputs, dim=3)
            movement_encoding = movement_encoding.view(bs, d, -1, h, w)

        if self.learnable:
            movement_encoding = movement_encoding.view(bs * d, -1, h, w)
            movement_encoding = self.conv(movement_encoding)
            movement_encoding = movement_encoding.view(bs, d, -1, h, w)

        if self.aggregation_type == 'sum':
            movement_encoding = movement_encoding.view(bs, d, num_kp, -1, h, w).sum(dim=2)

        return movement_encoding.permute(0, 2, 1, 3, 4)


if __name__ == "__main__":
    import imageio
    import numpy as np

    kp_array = np.zeros((2, 1, 6, 2), dtype='float32')
    kp_var = np.zeros((2, 1, 6, 2, 2), dtype='float32')

    for i in range(5, 11):
        kp_array[0, :, i - 5, 0] = 3 * i / 64
        kp_array[0, :, i - 5, 1] = 6 * i / 128

        kp_array[1, :, i - 5, 0] = 3 * i / 64
        kp_array[1, :, i - 5, 1] = 6 * i / 128

    kp_var[:, :, :, 0, 0] = 0.01
    kp_var[:, :, :, 0, 1] = 0.005
    kp_var[:, :, :, 1, 0] = 0.005
    kp_var[:, :, :, 1, 1] = 0.03

    kp_array = 2 * kp_array - 1

    mean_kp = torch.from_numpy(kp_array)
    var_kp = torch.from_numpy(kp_var)

    out = kp2gaussian({'mean' : mean_kp, 'var': var_kp}, (128, 64))

    kp = gaussian2kp(out)

    print (kp['mean'][0])
    print (kp['var'][0])

    out = out.numpy()

    out /= out.max()
    out = 1 - np.squeeze(out)
    imageio.mimsave('movie1.gif', out[0])
    imageio.mimsave('movie2.gif', out[1])
