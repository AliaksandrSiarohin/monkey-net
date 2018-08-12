from argparse import ArgumentParser
import os
import yaml
from time import gmtime, strftime
from tqdm import trange

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from frames_dataset import VideoToTensor, NormalizeKP, FramesDataset
from logger import Logger
from modules.dd_model import DDModel
from modules.losses import total_loss


def train(config, checkpoint, log_dir, device_ids):
    start_iter = 0
    model = DDModel(block_expansion=config['block_expansion'],
                    spatial_size=config['spatial_size'],
                    num_channels=config['num_channels'],
                    num_kp=config['num_kp'],
                    kp_gaussian_sigma=config['kp_gaussian_sigma'],
                    deformation_type=config['deformation_type'])

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

    if checkpoint is not None:
        start_iter = Logger.load_cpk(checkpoint, model, optimizer)

    with Logger(model=model, optimizer=optimizer, log_dir=log_dir,
                log_freq=config['log_freq'], cpk_freq=config['cpk_freq']) as logger:
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

            loss, loss_list = total_loss(x, out, config['loss_weights'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.log(it, loss_list=loss_list, out=out, inp=x)


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

    log_dir = os.path.join(opt.log_dir, opt.config.split('.')[0] + ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime()))

    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.mode == 'train':
        print ("Start model training...")
        train(config, opt.checkpoint, log_dir, opt.device_ids)
    elif opt.mode == 'test':
        print ("Start model testing...")
        test(config, opt.checkpoint)
    elif opt.mode == 'transfer':
        print ("Transfering motion...")
        transfer(config, opt.checkpoint)

