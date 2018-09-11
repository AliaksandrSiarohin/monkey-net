import torch
import torch.nn.functional as F
import numpy as np

from modules.util import make_coordinate_grid, compute_image_gradient


def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * torch.mean(torch.abs(prediction - target))


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


def grad_reconstruction(deformation, target, weight):
    if weight == 0:
        return 0
    target = target.permute(0, 2, 1, 3, 4).contiguous()
    bs, d, c, h, w = target.shape

    target_first_frame = target[:, 0, :, :, :]

    grad = compute_image_gradient(target_first_frame, padding=1)
    grad = grad.unsqueeze(2)
    deformed_grad = F.grid_sample(grad, deformation)

    true_grad = compute_image_gradient(target.view(bs * d, c, h, w), padding=1)
    true_grad = true_grad.view(bs, d, -1, h, w).permute(0, 2, 1, 3, 4)
    return reconstruction_loss(deformed_grad, true_grad, weight)


def flow_reconstruction(deformation, flow, weight):
    if weight == 0 or flow is None:
        return 0

    deformation = deformation[..., :2]
    bs, d, h, w, _ = deformation.shape
    deformation = deformation.view(bs, d, h, w, 2)
    grid = make_coordinate_grid((h, w), deformation.type())
    grid = grid.unsqueeze(0).unsqueeze(0)

    deformation_relative = (deformation - grid)

    return reconstruction_loss(deformation_relative, flow, weight)


def kp_movement_loss(kp_video, deformation, weight):
    kp_video = kp_video.detach()
    if weight == 0:
        return 0
    deformation = deformation[..., :2]
    bs, d, h, w, _ = deformation.shape
    deformation = deformation.view(-1, h * w, 2)
    bs, d, num_kp, _ = kp_video.shape

    kp_index = ((kp_video.view(-1, num_kp, 2) + 1) / 2)
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


def kp_stationary_loss(kp_video, allowed_movement, weight):
    kp_video_diff = torch.cat([kp_video[:, 0].unsqueeze(1), kp_video[:, :-1]], dim=1) - kp_video
    kp_video_diff = torch.abs(kp_video_diff)
    penalty = torch.max(torch.zeros_like(kp_video_diff), kp_video_diff - allowed_movement)
    return weight * penalty.mean()


def total_loss(inp, out, loss_weights):
    video_gt = inp['video_array']

    video_prediction = out['video_prediction']
    video_deformed = out['video_deformed']
    kp = out['kp_array']
    deformation = out['deformation']

    if video_prediction.type() != video_gt.type():
        video_gt = video_gt.type(video_prediction.type())

    if 'flow_array' in inp:
        flow = inp['flow_array']
        if flow.type() != deformation.type():
            flow = flow.type(deformation.type())
    else:
        flow = None

    loss_names = ["reconstruction", "reconstruction_deformed", "kp_movement", "tv", "reconstruction_grad",
                  "flow_reconstruction", "kp_stationary"]
    loss_values = [reconstruction_loss(video_prediction, video_gt, loss_weights["reconstruction"]),
                   reconstruction_loss(video_deformed,   video_gt, loss_weights["reconstruction_deformed"]),
                   kp_movement_loss(kp, deformation, loss_weights["kp_movement"]),
                   tv_loss(deformation, video_gt, loss_weights["tv"], loss_weights['tv_border']),
                   grad_reconstruction(deformation, video_gt, loss_weights["reconstruction_grad"]),
                   flow_reconstruction(deformation, flow, loss_weights['reconstruction_flow']),
                   kp_stationary_loss(kp, loss_weights['allowed_movement'], loss_weights['kp_stationary'])]

    total = sum(loss_values)

    loss_values.append(total)
    loss_names.append("total")

    loss_values = [0 if type(value) == int else value.detach().cpu().numpy() for value in loss_values]

    return total, zip(loss_names, loss_values)







