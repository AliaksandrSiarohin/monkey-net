from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, Encoder, SameBlock3D
from modules.movement_embedding import MovementEmbeddingModule


class AffineDeformation(nn.Module):
    """
    Deformation module receive keypoint cordinates and try to predict values for affine deformation. It has MPL architecture.
    """
    def __init__(self, num_kp, num_blocks=3):
        super(AffineDeformation, self).__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(nn.Conv1d((2 ** i) * 2 * num_kp, ((2 ** (i + 1)) * 2 * num_kp if i != num_blocks - 1 else 6),
                                    kernel_size=1))

        self.blocks = nn.ModuleList(blocks)
        self.blocks[-1].weight.data.zero_()
        self.blocks[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, difference_embedding):
        bs, c, d, h, w = difference_embedding.shape

        out = difference_embedding[..., 0, 0]

        for block in self.blocks:
            out = block(out)
            out = F.leaky_relu(out, 0.2)
        out = out.view(-1, 2, 3)

        deformations_absolute = F.affine_grid(out, torch.Size((out.shape[0], 3, h, w)))
        deformations_absolute = deformations_absolute.view((bs, 1, d, h, w, -1))

        return deformations_absolute.permute(0, 1, 5, 2, 3, 4)


class DeformationModule(nn.Module):
    """
    Deformation module, predict deformation, that will be applied to skip connections.
    """
    def __init__(self, block_expansion, num_blocks, max_features, mask_embedding_params, num_kp,
                 num_channels, kp_variance, num_group_blocks, use_correction, use_mask, camera_params=None):
        super(DeformationModule, self).__init__()
        self.mask_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels,
                                                      **mask_embedding_params)
        self.difference_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels,
                                                            use_difference=True, use_heatmap=False, use_deformed_appearance=False)

        group_blocks = []
        for i in range(num_group_blocks):
            group_blocks.append(SameBlock3D(self.mask_embedding.out_channels, self.mask_embedding.out_channels,
                                            groups=num_kp, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        self.group_blocks = nn.ModuleList(group_blocks)

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=self.mask_embedding.out_channels,
                                   out_features=(num_kp + 1) * use_mask + 2 * use_correction,
                                   max_features=max_features, dim=3, num_blocks=num_blocks)
        self.hourglass.decoder.conv.weight.data.zero_()
        bias_init = ([2] + [0] * num_kp) * use_mask + [0, 0] * use_correction
        self.hourglass.decoder.conv.bias.data.copy_(torch.tensor(bias_init, dtype=torch.float))

        if camera_params is not None:
            self.camera_module = AffineDeformation(num_kp=num_kp, **camera_params)
        else:
            self.camera_module = None
        self.num_kp = num_kp
        self.use_correction = use_correction
        self.use_mask = use_mask

    def forward(self, kp_video, appearance_frame):
        prediction = self.mask_embedding(kp_video, appearance_frame)
        for block in self.group_blocks:
            prediction = block(prediction)
            prediction = F.leaky_relu(prediction, 0.2)
        prediction = self.hourglass(prediction)
        bs, _, d, h, w = prediction.shape
        if self.use_mask:
            mask = prediction[:, :(self.num_kp + 1)]
            mask = F.softmax(mask, dim=1)
            mask = mask.unsqueeze(2)
            difference_embedding = self.difference_embedding(kp_video, appearance_frame)
            if self.camera_module is None:
                shape = (bs, 1, 2, d, h, w)
                camera_prediction = torch.zeros(shape).type(difference_embedding.type())
            else:
                camera_prediction = self.camera_module(difference_embedding)

            difference_embedding = torch.cat([camera_prediction, difference_embedding.view(bs, self.num_kp, 2, d, h, w)], dim=1)
            deformations_relative = (difference_embedding * mask).sum(dim=1)
        else:
            deformations_relative = 0

        if self.use_correction:
            correction = prediction[:, -2:]
        else:
            correction = 0

        deformations_relative = deformations_relative + correction
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
        deformation = deformations_relative + coordinate_grid
        z_coordinate = torch.zeros(deformation.shape[:-1] + (1, )).type(deformation.type())

        return torch.cat([deformation, z_coordinate], dim=-1)



class IdentityDeformation(nn.Module):
    def forward(self, kp_appearance, kp_video, appearance_frame):
        bs, _, _, h, w = appearance_frame.shape
        _, d, num_kp, _ = kp_video['mean'].shape
        coordinate_grid = make_coordinate_grid((h, w), type=appearance_frame.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)

        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1, )).type(coordinate_grid.type())
        return torch.cat([coordinate_grid, z_coordinate], dim=-1)
