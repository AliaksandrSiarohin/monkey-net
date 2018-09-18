import torch
import torch.nn.functional as F
import numpy as np

from modules.util import make_coordinate_grid, compute_image_gradient


def tv_loss(deformation, target, loss_weight, border_weight=1):
    if loss_weight == 0:
        return 0
    deformation = deformation[..., :2]
    target = target.permute(0, 2, 1, 3, 4).contiguous()
    bs, d, c, h, w = target.shape
    target = target.view(bs * d, c, h, w)

    border = compute_image_gradient(target).abs().sum(dim=1)
    border = torch.exp(-border_weight * border)

    deformation = deformation.view(bs * d, h, w, 2)
    grid = make_coordinate_grid((h, w), deformation.type())
    grid = grid.unsqueeze(0)

    deformation_relative = (deformation - grid)
    deformation_relative = deformation_relative.permute(0, 3, 1, 2)

    deformation_grad = compute_image_gradient(deformation_relative).abs().sum(dim=1)

    loss = border * deformation_grad

    return torch.mean(loss) * loss_weight


def kp_movement_loss(deformation, kp_video, weight):
    kp_video = kp_video['mean']
    if weight == 0:
        return 0

    deformation = deformation[..., :2]

    bs, d, h, w, _ = deformation.shape
    deformation = deformation.view(-1, h * w, 2)
    bs, d, num_kp, _ = kp_video.shape

    kp_index = ((kp_video.contiguous().view(-1, num_kp, 2) + 1) / 2)
    multiple = torch.from_numpy(np.array((w - 1, h - 1))).view(1, 1, 2).type(kp_index.type())
    kp_index = multiple * kp_index

    kp_index = kp_index.long()
    kp_index = kp_index[:, :, 0] + kp_index[:, :, 1] * w

    kp_values_x = torch.gather(deformation[..., 0], dim=1, index=kp_index)
    kp_values_y = torch.gather(deformation[..., 1], dim=1, index=kp_index)

    kp_values_x = kp_values_x.view(bs, d, num_kp)
    kp_values_y = kp_values_y.view(bs, d, num_kp)

    target = kp_video[:, 0, :, :].view(bs, 1, num_kp, 2)

    y_loss = torch.mean(torch.abs(target[..., 1] - kp_values_y))
    x_loss = torch.mean(torch.abs(target[..., 0] - kp_values_x))

    total = y_loss + x_loss

    return weight * total


def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * torch.mean(torch.abs(prediction - target))


def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    return weight * score.mean()


def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real[-1]
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2
    return weight * score.mean()


def generator_loss(discriminator_maps_generated, discriminator_maps_real, discriminator_maps_deformed,
                   deformation, kp_video, loss_weights):
    loss_names = []
    loss_values = []
    if loss_weights['reconstruction_deformed'] is not None:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_deformed[:-1])):
            loss_names.append("layer-%s_rec_def" % i)
            loss_values.append(reconstruction_loss(b, a, weight=loss_weights['reconstruction_deformed'][i]))

    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            loss_names.append("layer-%s_rec" % i)
            loss_values.append(reconstruction_loss(b, a, weight=loss_weights['reconstruction'][i]))


    loss_names.append("gen_gan")
    loss_values.append(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))

    loss_names.append('tv')
    loss_values.append(tv_loss(deformation, discriminator_maps_real[0], loss_weights['tv'], loss_weights['tv_border']))

    loss_names.append('kp_movement')
    loss_values.append(kp_movement_loss(deformation, kp_video, loss_weights['kp_movement']))

    total = sum(loss_values)

    loss_values.append(total)
    loss_names.append("total")

    loss_values = [0 if type(value) == int else value.detach().cpu().numpy() for value in loss_values]

    return total, loss_names, loss_values


def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_names = ['disc_gan']
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real,
                                          loss_weights['discriminator_gan'])]

    total = sum(loss_values)
    loss_values = [0 if type(value) == int else value.detach().cpu().numpy() for value in loss_values]

    return total, loss_names, loss_values



