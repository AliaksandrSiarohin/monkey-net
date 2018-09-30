import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from modules.util import matrix_inverse


def combine_kp(kp_appearance, kp_video, scale_difference=False):
    kp_video_diff = kp_video['mean'] - kp_video['mean'][:, 0:1]

    if scale_difference:
        _, app_coef = compute_pairwise_distances(kp_appearance['mean'])
        _, video_coef = compute_pairwise_distances(kp_video['mean'][:, 0:1])
        mult = app_coef / video_coef
        mult = mult.type(kp_video_diff.type())
        kp_video_diff *= mult

    kp_mean = kp_video_diff + kp_appearance['mean']
    out = {'mean': kp_mean}

    if 'var' in kp_video:
        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1]))
        kp_var = torch.matmul(kp_var, kp_appearance['var'])
        out['var'] = kp_var

    return out


def compute_pairwise_distances(kp_array):
    bs, d, num_kp, _ = kp_array.shape

#    distances = torch.zeros(bs, d, num_kp, num_kp)

#    for i in range(num_kp):
#        for j in range(num_kp):
#            distances[:, :, i, j] = torch.abs(kp_array[:, :, i] - kp_array[:, :, j]).sum(dim=-1)

    center_of_mass = kp_array.mean(dim=2, keepdim=True)

#    distances = distances.view(bs, d, -1)
#    median =  torch.sqrt(distances.var(dim=-1, keepdim=True))#[0]
#    distances /= median

    distances = kp_array - center_of_mass

    return distances, distances.var()


def select_best_frame(kp_video, kp_appearance):
    video_distances, _ = compute_pairwise_distances(kp_video)
    appearance_distances, _ = compute_pairwise_distances(kp_appearance)

    norm = torch.abs(video_distances - appearance_distances).sum(dim=-1).sum(dim=-1)

    best_frame = torch.argmin(norm, dim=-1)
    return best_frame.squeeze(dim=0)


def transfer(config, generator, kp_extractor, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'transfer')
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=100)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_extractor=kp_extractor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    generator = generator.module
    transfer_params = config['transfer_params']
    generator.eval()
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value.cuda() for key,value in x.items()}

            motion_video = x['first_video_array']
            appearance_frame = x['second_video_array'][:, :, :1, :, :]

            kp_video = kp_extractor(motion_video)
            kp_appearance = kp_extractor(appearance_frame)

            if transfer_params['select_best_frame']:
                best_frame = select_best_frame(kp_video['mean'], kp_appearance['mean'])
            else:
                best_frame = 0

            reverse_sample = range(0, best_frame + 1)[::-1]
            first_video_seq = {k: v[:, reverse_sample] for k, v in kp_video.items()}
            first_out = generator(appearance_frame, combine_kp(kp_appearance, first_video_seq,
                                                               transfer_params['scale_difference']))

            second_video_seq = {k: v[:, best_frame:] for k, v in kp_video.items()}
            second_out = generator(appearance_frame, combine_kp(kp_appearance, second_video_seq,
                                                                transfer_params['scale_difference']))

            out = dict()

            sample = range(1, best_frame + 1)[::-1]
            out['video_prediction'] = torch.cat([first_out['video_prediction'][:, :, sample],
                                                 second_out['video_prediction']], dim=2)
            out['video_deformed'] = torch.cat([first_out['video_deformed'][:, :, sample],
                                               second_out['video_deformed']], dim=2)

            out['kp_video'] = kp_video
            out['kp_appearance'] = kp_appearance
            d = motion_video.shape[2]
            out['best_frame'] = motion_video[:, :, best_frame:(best_frame + 1)].repeat(1, 1, d, 1, 1)
            out['best_kp'] = {k: v[:, best_frame:(best_frame + 1)].repeat(1, d, 1, 1) for k, v in kp_video.items()}

            image = Visualizer().visualize_transfer(inp=x, out=out)
            imageio.mimsave(os.path.join(log_dir, str(it).zfill(8) + '.gif'), image)
