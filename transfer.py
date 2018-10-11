import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from modules.util import matrix_inverse


def normalize_kp(kp_video, kp_appearance):
    kp_video = {k: v for k, v in kp_video.items()}
    kp_video['mean'] = (kp_video['mean'] - kp_video['mean'][:, 0:1])
    kp_video['mean'] = kp_video['mean']  + kp_appearance['mean']

    if 'var' in kp_video:
        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1]))
        kp_var = torch.matmul(kp_var, kp_appearance['var'])
        kp_video['var'] = kp_var
    
    return kp_video


def compute_center_scale(kp_array):
    bs, d, num_kp, _ = kp_array.shape

    center = kp_array.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    distances = kp_array - center
    var = (distances ** 2).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)
    sd = torch.sqrt(var)

    return distances/ sd, center, sd


def transfer(config, generator, kp_extractor, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'transfer')
    transfer_params = config['transfer_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=transfer_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_extractor=kp_extractor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    generator = generator.module

    generator.eval()
    kp_extractor.eval()
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value if not hasattr(value, 'cuda') else value.cuda() for key,value in x.items()}

            motion_video = x['first_video_array']
            appearance_frame = x['second_video_array'][:, :, :1, :, :]

            kp_video = kp_extractor(motion_video)
            kp_appearance = kp_extractor(appearance_frame)
            kp_video_norm = normalize_kp(kp_video, kp_appearance)
            out = generator(appearance_frame=appearance_frame, kp_video=kp_video_norm, kp_appearance=kp_appearance)
            out['kp_video'] = kp_video
            out['kp_appearance'] = kp_appearance

            image = Visualizer().visualize_transfer(inp=x, out=out)

            img_name = "-".join([x['first_name'][0], x['second_name'][0]]) + transfer_params['format']
            imageio.mimsave(os.path.join(log_dir, img_name), image)
