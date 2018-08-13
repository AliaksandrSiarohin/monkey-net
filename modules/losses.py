import torch


def reconstruction_loss(prediction, target, weight):
    return weight * torch.mean(torch.abs(prediction - target))


def kp_movement_loss(kp_video, deformation, weight):
    bs, d, h, w, _ = deformation.shape
    deformation = deformation.view(-1, h * w, 2)
    bs, d, num_kp, _ = kp_video.shape

    kp_index = h * ((kp_video.view(-1, num_kp, 2) + 1) / 2)
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


def total_loss(inp, out, loss_weights):
    video_gt = inp['video_array']
    video_prediction = out['video_prediction']
    if video_prediction.type() != video_gt.type():
        video_gt = video_gt.type(video_prediction.type())

    video_deformed = out['video_deformed']
    kp = out['kp_array']
    deformation = out['deformation']

    loss_names = ["reconstruction", "reconstruction_deformed", "kp_movement"]
    loss_values = [reconstruction_loss(video_prediction, video_gt, loss_weights["reconstruction"]),
                   reconstruction_loss(video_deformed,   video_gt, loss_weights["reconstruction_deformed"]),
                   kp_movement_loss(kp, deformation, loss_weights["kp_movement"])]

    total = sum(loss_values)

    loss_values.append(total)
    loss_names.append("total")

    loss_values = [value.detach().cpu().numpy() for value in loss_values]

    return total, zip(loss_names, loss_values)







