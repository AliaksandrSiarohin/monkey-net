import torch
import torch.nn.functional as F
import numpy as np

from modules.util import compute_image_gradient


def resize_input(inp, channel_dim, scale):
    if channel_dim != 1:
        per_range = list(range(len(inp.shape)))
        del per_range[channel_dim]
        per_range.insert(1, channel_dim)
        per_range_inv = [per_range.index(l) for l in range(len(per_range))]
        inp = inp.permute(*per_range)

    out = F.interpolate(inp, scale_factor=scale)

    if channel_dim != 1:
        out = out.permute(*per_range_inv)

    return out


def resize_kp(kp, scale):
    bs, d, num_kp, _ = kp.shape
    kp_view = kp.permute(0, 2, 3, 1).view(bs, -1, d)
    kp_interpolated = F.interpolate(kp_view, scale_factor=scale, mode='nearest')
    kp_interpolated = kp_interpolated.view(bs, num_kp, 2, int(d * scale)).permute(0, 3, 1, 2)
    return kp_interpolated


def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * torch.mean(torch.abs(prediction - target))


def reconstruction_loss_deformed(deformation, target, weight):
    if weight == 0:
        return 0
    target = resize_input(target, channel_dim=1, scale=deformation.shape[1] / target.shape[2])
    prediction = F.grid_sample(target, deformation)

    return reconstruction_loss(prediction, target, weight)


def tv_loss(deformation_relative, target, loss_weight, border_weight=1):
    if loss_weight == 0:
        return 0

    target = resize_input(target, channel_dim=1, scale=deformation_relative.shape[1] / target.shape[2])
    target = target.permute(0, 2, 1, 3, 4).contiguous()
    bs, d, c, h, w = target.shape
    target = target.view(bs * d, c, h, w)

    border = compute_image_gradient(target).abs().sum(dim=1)
    border = torch.exp(-border_weight * border)

    deformation_relative = deformation_relative.view(bs * d, h, w, 2)
    deformation_relative = deformation_relative.permute(0, 3, 1, 2)

    deformation_grad = compute_image_gradient(deformation_relative).abs().sum(dim=1)

    loss = border * deformation_grad

    return torch.mean(loss) * loss_weight


def grad_reconstruction(deformation, target, weight):
    if weight == 0:
        return 0

    target = resize_input(target, 1, deformation.shape[1] / target.shape[2])

    target = target.permute(0, 2, 1, 3, 4)
    bs, d, c, h, w = target.shape

    grad = compute_image_gradient(target.view(bs * d, c, h, w), padding=1)
    grad = grad.view(bs, d, c, h, w).contiguous().permute(0, 2, 1, 3, 4)

    deformed_grad = F.grid_sample(grad, deformation)

    return reconstruction_loss(deformed_grad, grad, deformed_grad)


def flow_reconstruction(deformation_relative, flow, weight):
    if weight == 0 or flow is None:
        return 0
    flow = resize_input(flow, 4, deformation_relative.shape[1] / flow.shape[1])
    return reconstruction_loss(deformation_relative, flow, weight)


def kp_movement_loss(deformation, kp_video, weight):
    kp_video = kp_video.detach()
    if weight == 0:
        return 0

    kp_video = resize_kp(kp_video,  deformation.shape[1] / kp_video.shape[1])
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


def compute_loss_for_all_inputs(gt, prediction, loss_fn, **kwargs):
    out = 0
    for pr in prediction:
        out += pr.shape[1] * loss_fn(pr, gt, **kwargs)
    return out


def total_loss(inp, out, loss_weights):
    video_gt = inp['video_array']

    video_prediction = out['video_prediction']
    kp = out['kp_array']

    if video_prediction.type() != video_gt.type():
        video_gt = video_gt.type(video_prediction.type())

    if 'flow_array' in inp:
        flow = inp['flow_array']
        if flow.type() != video_prediction.type():
            flow = flow.type(video_prediction.type())
    else:
        flow = None

    loss_names = ["reconstruction", "reconstruction_deformed", "kp_movement", "tv", "reconstruction_grad",
                  "flow_reconstruction"]
    loss_values = [reconstruction_loss(video_prediction, video_gt, loss_weights["reconstruction"]),
                   compute_loss_for_all_inputs(video_gt, out['absolute_to_previous'], reconstruction_loss_deformed,
                                               weight=loss_weights["reconstruction_deformed"]),
                   compute_loss_for_all_inputs(kp, out['absolute_to_first'], kp_movement_loss,
                                               weight=loss_weights["kp_movement"]),
                   compute_loss_for_all_inputs(video_gt, out['relative_to_first'], tv_loss,
                                               loss_weight=loss_weights["tv"], border_weight=loss_weights["tv_border"]),
                   compute_loss_for_all_inputs(video_gt, out['absolute_to_previous'], grad_reconstruction,
                                               weight=loss_weights["reconstruction_grad"]),
                   compute_loss_for_all_inputs(flow, out['relative_to_first'], flow_reconstruction,
                                               weight=loss_weights["reconstruction_flow"])]

    total = sum(loss_values)

    loss_values.append(total)
    loss_names.append("total")

    loss_values = [0 if type(value) == int else value.detach().cpu().numpy() for value in loss_values]

    return total, zip(loss_names, loss_values)







