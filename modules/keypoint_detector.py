from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, matrix_inverse


def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def gaussian2kp(heatmap, kp_variance='matrix'):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

    mean = (heatmap * grid).sum(dim=(3, 4))

    kp = {'mean': mean.permute(0, 2, 1, 3)}

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(3, 4))
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(3, 4))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    return kp


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and variance.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 kp_variance, scale_factor=1):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, num_blocks=num_blocks)
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor != 1:
           x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        heatmap = self.predictor(x)
        final_shape = heatmap.shape
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=3)
        heatmap = heatmap.view(*final_shape)

        out = gaussian2kp(heatmap, self.kp_variance)

        return out


if __name__ == "__main__":
    import imageio
    import numpy as np

    kp_array = np.zeros((2, 1, 6, 2), dtype='float32')
    kp_var = np.zeros((2, 1, 6, 1, 1), dtype='float32')

    for i in range(5, 11):
        kp_array[0, :, i - 5, 0] = 3 * i / 64
        kp_array[0, :, i - 5, 1] = 6 * i / 128

        kp_array[1, :, i - 5, 0] = 3 * i / 64
        kp_array[1, :, i - 5, 1] = 6 * i / 128

    # kp_var[:, :, :, 0, 0] = 0.01
    # kp_var[:, :, :, 0, 1] = 0.005
    # kp_var[:, :, :, 1, 0] = 0.005
    # kp_var[:, :, :, 1, 1] = 0.03

    kp_var[:, :, :, 0] = 0.01

    kp_array = 2 * kp_array - 1

    mean_kp = torch.from_numpy(kp_array)
    var_kp = torch.from_numpy(kp_var)

    out = kp2gaussian({'mean': mean_kp, 'var': var_kp}, (128, 128), kp_variance=0.01)  # 'single')

    kp = gaussian2kp(out, kp_variance='single')

    print(kp['var'].shape)
    print(kp['mean'][0])
    print(kp['var'][0])

    out = out.numpy()

    out /= out.max()
    out = 1 - np.squeeze(out)
    imageio.mimsave('movie1.gif', out[0])
    imageio.mimsave('movie2.gif', out[1])
