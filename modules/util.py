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
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
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
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """
    def __init__(self, block_expansion, in_features, number_of_blocks=3, max_features=256, dim=2):
        super(Encoder, self).__init__()

        down_blocks = []

        kernel_size = (3, 3, 3) if dim == 3 else (1, 3, 3)
        padding = (1, 1, 1) if dim == 3 else (0, 1, 1)
        for i in range(number_of_blocks):
            down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]

        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """
    def __init__(self, block_expansion, in_features, out_features, number_of_blocks=3, max_features=256, dim=2,
                 additional_features_for_block=0):
        super(Decoder, self).__init__()
        kernel_size = (3, 3, 3) if dim == 3 else (1, 3, 3)
        padding = (1, 1, 1) if dim == 3 else (0, 1, 1)

        up_blocks = []

        for i in range(number_of_blocks)[::-1]:
            up_blocks.append(UpBlock3D((1 if i == number_of_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1))) + additional_features_for_block,
                                       min(max_features, block_expansion * (2 ** i)),
                                       kernel_size=kernel_size, padding=padding))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv = nn.Conv3d(in_channels=block_expansion + in_features + additional_features_for_block,
                              out_channels=out_features, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)
        return self.conv(out)


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """
    def __init__(self, block_expansion, in_features, out_features, number_of_blocks=3, max_features=256, dim=2):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, number_of_blocks, max_features, dim)
        self.decoder = Decoder(block_expansion, in_features, out_features, number_of_blocks, max_features, dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
