import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
from modules.losses import reconstruction_loss
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def generate(generator, appearance_image, kp_appearance, kp_video):
    out = {'video_prediction': [], 'video_deformed': []}
    for i in range(kp_video['mean'].shape[1]):
        kp_target = {k: v[:, i:(i + 1)] for k, v in kp_video.items()}
        kp_dict_part = {'kp_driving': kp_target, 'kp_source': kp_appearance}
        out_part = generator(appearance_image, **kp_dict_part)
        out['video_prediction'].append(out_part['video_prediction'])
        out['video_deformed'].append(out_part['video_deformed'])

    out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
    out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)
    out['kp_driving'] = kp_video
    out['kp_source'] = kp_appearance
    return out


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            kp_appearance = kp_detector(x['video'][:, :, :1])
            d = x['video'].shape[2]
            kp_video = cat_dict([kp_detector(x['video'][:, :, i:(i + 1)]) for i in range(d)], dim=1)

            out = generate(generator, appearance_image=x['video'][:, :, :1], kp_appearance=kp_appearance,
                           kp_video=kp_video)
            x['source'] = x['video'][:, :, :1]

            # Store to .png for evaluation
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))

            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)

            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video'].cpu(), 1)
            loss_list.append(loss.data.cpu().numpy())
            del x, kp_video, kp_appearance, out, loss
    print("Reconstruction loss: %s" % np.mean(loss_list))
