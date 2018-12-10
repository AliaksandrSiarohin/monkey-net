import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from modules.util import matrix_inverse
from scipy.spatial import ConvexHull
import numpy as np

from sync_batchnorm import DataParallelWithCallback


def make_symetric_matrix(torch_matrix):
    a = torch_matrix.cpu().numpy()
    c = (a + np.transpose(a, (0, 1, 2, 4, 3))) / 2
    d, u = np.linalg.eig(c)
    d[d <= 0] = 1e-6
    d_matrix = np.zeros_like(a)
    d_matrix[..., 0, 0] = d[..., 0]
    d_matrix[..., 1, 1] = d[..., 1]
    res = np.matmul(np.matmul(u, d_matrix), np.transpose(u, (0, 1, 2, 4, 3)))
    res = torch.from_numpy(res).type(torch_matrix.type())

    return res


def normalize_kp(kp_video, kp_appearance, movement_mult=False, move_location=False, adapt_variance=False, clip_mean=False):
    if movement_mult:
        appearance_area = ConvexHull(kp_appearance['mean'][0, 0].data.cpu().numpy()).volume
        video_area = ConvexHull(kp_video['mean'][0, 0].data.cpu().numpy()).volume
        movement_mult = np.sqrt(appearance_area) / np.sqrt(video_area)
    else:
        movement_mult = 1

    kp_video = {k: v for k, v in kp_video.items()}

    if move_location:
        kp_video_diff = (kp_video['mean'] - kp_video['mean'][:, 0:1])
        kp_video_diff *= movement_mult
        kp_video['mean'] = kp_video_diff + kp_appearance['mean']

    if clip_mean:
        one = torch.ones(1).type(kp_video_diff.type())
        kp_video['mean'] = torch.max(kp_video['mean'], -one)
        kp_video['mean'] = torch.min(kp_video['mean'], one)

    if ('var' in kp_video) and adapt_variance:
        var_first = kp_video['var'][:, 0:1].repeat(1, kp_video['var'].shape[1], 1, 1, 1)
        kp_var, _ = torch.gesv(var_first, kp_video['var'])

        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1], eps=0))
        kp_var = torch.matmul(kp_var, kp_appearance['var'])

        kp_var = make_symetric_matrix(kp_var)
        kp_video['var'] = kp_var

    return kp_video


def transfer(config, generator, kp_extractor, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'transfer')
    png_dir = os.path.join(log_dir, 'png')
    transfer_params = config['transfer_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=transfer_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_extractor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    generator = DataParallelWithCallback(generator)
    kp_extractor = DataParallelWithCallback(kp_extractor)

    generator.eval()
    kp_extractor.eval()
    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value if not hasattr(value, 'cuda') else value.cuda() for key, value in x.items()}
            motion_video = x['first_video_array']
            d = motion_video.shape[2]
            appearance_frame = x['second_video_array'][:, :, :1, :, :]
            kp_video = cat_dict([kp_extractor(motion_video[:, :, i:(i + 1)]) for i in range(d)], dim=1)
            kp_appearance = kp_extractor(appearance_frame)

            kp_video_norm = normalize_kp(kp_video, kp_appearance, **transfer_params['normalization_params'])
            kp_video_list = [{k: v[:, i:(i + 1)] for k, v in kp_video_norm.items()} for i in range(d)]
            out = cat_dict([generator(appearance_frame=appearance_frame, kp_video=kp, kp_appearance=kp_appearance)
                            for kp in kp_video_list], dim=2)
            out['kp_video'] = kp_video
            out['kp_appearance'] = kp_appearance
            out['kp_norm'] = kp_video_norm

            img_name = "-".join([x['first_name'][0], x['second_name'][0]])

            # Store to .png for evaluation
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, img_name + '.png'), (255 * out_video_batch).astype(np.uint8))

            image = Visualizer(**config['visualizer_params']).visualize_transfer(inp=x, out=out)

            imageio.mimsave(os.path.join(log_dir, img_name + transfer_params['format']), image)
