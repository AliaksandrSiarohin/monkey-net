from torch import nn

import torch.nn.functional as F
import torch

import numpy as np


def compute_image_gradient(image, padding=0):
    bs, c, h, w = image.shape

    sobel_x = torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])).type(image.type())
    filter = sobel_x.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(image, filter, groups=c, padding=padding)
    grad_x = grad_x

    sobel_y = torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])).type(image.type())
    filter = sobel_y.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_y = F.conv2d(image, filter, groups=c, padding=padding)
    grad_y = grad_y

    return torch.cat([grad_x, grad_y], dim=1)


def make_coordinate_grid(spatial_size, type):
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

class DownBlock2D(nn.Module):
    """
    Simple block for processing each frame separately (encoder).
    """
    def __init__(self, in_features, out_features):
        super(DownBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class UpBlock2D(nn.Module):
    """
    Simple block for processing each frame separately (decoder).
    """
    def __init__(self, in_features, out_features):
        super(UpBlock2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """
    def __init__(self, in_features, out_features):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out



class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self, in_features, out_features):
        super(DownBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


def kp2gaussian(kp, spatial_size, sigma):
    coordinate_grid = make_coordinate_grid(spatial_size, kp.type())

    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1, ) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)

    # Computing gaussian
    squares = (coordinate_grid - kp) ** 2
    sum = torch.sum(squares, dim=-1)
    out = torch.exp(-sum / (2 * sigma ** 2))

    return out

# class KP2Gaussian(nn.Module):
#     """
#     Represent normalized[0, 1] coordinates keypoints as feature map
#     """
#     def __init__(self, spatial_size, sigma):
#         super(KP2Gaussian, self).__init__()
#         self.spatial_size = spatial_size
#         self.sigma = sigma
#
#     def forward(self, kp):
#         # Preprocess coordinate grid
#         coordinate_grid = make_coordinate_grid(self.spatial_size, kp.type())
#
#         number_of_leading_dimensions = len(kp.shape) - 1
#         shape = (1, ) * number_of_leading_dimensions + coordinate_grid.shape
#
#         coordinate_grid = coordinate_grid.view(*shape)
#         repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
#         coordinate_grid = coordinate_grid.repeat(*repeats)
#
#         # Preprocess kp shape
#         shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
#         kp = kp.view(*shape)
#
#         # Computing gaussian
#         squares = (coordinate_grid - kp) ** 2
#         sum = torch.sum(squares, dim=-1)
#         out = torch.exp(-sum / (2 * self.sigma ** 2))
#
#         return out


if __name__ == "__main__":
    import imageio

    kp_array = np.zeros((2, 1, 64, 2), dtype='float32')

    for i in range(10):
        kp_array[0, :, i, 0] = 6 * i / 64
        kp_array[0, :, i, 1] = 1 * i / 128

        kp_array[1, :, i, 0] = 1 * i / 64
        kp_array[1, :, i, 1] = 12 * i / 128

    kp_array = 2 * kp_array - 1

    tkp = torch.from_numpy(kp_array)

    out = kp2gaussian(tkp, (128, 64), 0.1)
    out = out.numpy()

    out = 1 - np.squeeze(out)
    imageio.mimsave('movie1.gif', out[0])
    imageio.mimsave('movie2.gif', out[1])
