import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio


def compute_pairwise_distances(kp_array):
    bs, d, num_kp, _ = kp_array.shape

    distances = torch.zeros(bs, d, num_kp, num_kp)

    for i in range(num_kp):
        for j in range(num_kp):
            distances[:, :, i, j] = torch.abs(kp_array[:, :, i] - kp_array[:, :, j]).sum(dim=-1)

    distances = distances.view(bs, d, -1)
    median = distances.median(dim=-1, keepdim=True)[0]
    distances /= median

    return distances


def select_best_frame(kp_video, kp_appearance):
    video_distances = compute_pairwise_distances(kp_video)
    appearance_distances = compute_pairwise_distances(kp_appearance)

    norm = torch.abs(video_distances - appearance_distances).sum(dim=-1)

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
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value.cuda() for key,value in x.items()}

            motion_video = x['first_video_array']
            appearance_frame = x['second_video_array'][:, :, :1, :, :]

            kp_video = kp_extractor(motion_video)
            kp_appearance = kp_extractor(appearance_frame)

            if config['transfer_params']['select_best_frame']:
                best_frame = select_best_frame(kp_video['mean'], kp_appearance['mean'])
            else:
                best_frame = 0

            reverse_sample = range(0, best_frame + 1)[::-1]
            first_video_seq = {k: v[:, reverse_sample] for k, v in kp_video.items()}
            first_out = generator(appearance_frame, first_video_seq, kp_appearance)

            second_video_seq = {k: v[:, best_frame:] for k, v in kp_video.items()}
            second_out = generator(appearance_frame, second_video_seq, kp_appearance)

            out = dict()

            sample = range(1, best_frame + 1)[::-1]
            out['video_prediction'] = torch.cat([first_out['video_prediction'][:, :, sample],
                                                 second_out['video_prediction']], dim=2)
            out['video_deformed'] = torch.cat([first_out['video_deformed'][:, :, sample],
                                               second_out['video_prediction']], dim=2)

            out['kp_video'] = kp_video
            out['kp_appearance'] = kp_appearance

            image = Visualizer().visualize_transfer(inp=x, out=out)
            imageio.mimsave(os.path.join(log_dir, str(it).zfill(8) + '.gif'), image)
