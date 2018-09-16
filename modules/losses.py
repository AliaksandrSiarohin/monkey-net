import torch
import torch.nn.functional as F
import numpy as np

from modules.util import make_coordinate_grid, compute_image_gradient


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


def generator_loss(discriminator_maps_generated, discriminator_maps_real, discriminator_maps_deformed, loss_weights):
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


