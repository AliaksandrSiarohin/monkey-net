from argparse import ArgumentParser
import yaml
import torch
from modules.dd_model import DDModel
from tqdm import trange

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from frames_dataset import VideoToTensor, NormalizeKP, FramesDataset

import os
import numpy as np


def train(config, checkpoint, log_folder, device_ids):
    start_iter = 0
    model = DDModel(block_expansion=config['block_expansion'],
                    spatial_size=config['spatial_size'],
                    num_channels=config['num_channels'],
                    num_kp=config['num_kp'],
                    kp_gaussian_sigma=config['kp_gaussian_sigma'])

    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    data_transform = transforms.Compose([
        VideoToTensor(),
        NormalizeKP(config['spatial_size'])
    ])

    dataset = FramesDataset(root_dir=config['data_dir'], transform=data_transform, offline_kp=config['offline_kp'],
                            image_shape=(config['spatial_size'], config['spatial_size'], 3))

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=config['data_load_workers'])

    if opt.checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter']

    log_file = open(os.path.join(log_folder, 'log.txt'), 'a')

    loss_list = []
    for it in trange(start_iter, config['num_iter']):
        x = next(iter(dataloader))

        motion_video = x['video_array']
        appearance_frame = x['video_array'][:, :, 0, :, :]

        if config['offline_kp']:
            kp_video = x['kp_array']
            kp_appearance = x['kp_array'][:, 0, :, :]
        else:
            kp_video, kp_appearance = None, None

        out = model(appearance_frame=appearance_frame, motion_video=motion_video,
                    kp_video=kp_video, kp_appearance=kp_appearance)

        target = x['video_array'].type(out.type())
        loss = torch.mean(torch.abs(target - out))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.detach().cpu().numpy())
        if it % config['log_freq'] == 0:
            print("Iterations %s, score: %s" % (it, np.mean(loss_list)), file=log_file)
            loss_list = []
            log_file.flush()

        if it % config['cpk_freq'] == 0:
            d = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": it}
            torch.save(d, os.path.join(log_folder, 'checkpoint%s.pth.tar' % str(it).zfill(8)))

    log_file.close()


def test(config, checkpoint):
    None

def transfer(config, checkpoint):
    None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "test", "transfer"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.mode == 'train':
        print ("Start model training...")
        train(config, opt.checkpoint, opt.log_dir, opt.device_ids)
    elif opt.mode == 'test':
        print ("Start model testing...")
        test(config, opt.checkpoint)
    elif opt.mode == 'transfer':
        print ("Transfering motion...")
        transfer(config, opt.checkpoint)

