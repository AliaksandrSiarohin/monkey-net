from argparse import ArgumentParser
import os
import yaml
from time import gmtime, strftime
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from frames_dataset import VideoToTensor, Normalize, FramesDataset, PairedDataset
from logger import Logger, Visualizer
from modules.dd_model import DDModel
from modules.losses import total_loss

import numpy as np
import imageio


def train(config, model, checkpoint, log_dir, dataset):
    start_iter = 0
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=config['data_load_workers'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    if checkpoint is not None:
        start_iter = Logger.load_cpk(checkpoint, model, optimizer)

    with Logger(model=model, optimizer=optimizer, log_dir=log_dir,
                log_freq=config['log_freq'], cpk_freq=config['cpk_freq']) as logger:
        for it in trange(start_iter, config['num_iter']):
            x = next(iter(dataloader))

            out = model(x)
            loss, loss_list = total_loss(x, out, config['loss_weights'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.log(it, loss_list=loss_list, inp=x)


def test(config, model, checkpoint, log_dir, dataset):
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model, None)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []
    for it, x in tqdm(enumerate(dataloader)):
        out = model(x)
        out = {key: value.data for key, value in out.items()}

        loss, losses = total_loss(x, out, config['loss_weights'])

        loss_names, values = list(zip(*losses))
        loss_list.append(values)

    print ("; ".join([name + " - " + str(value) for name, value in zip(loss_names, np.array(loss_list).mean(axis=0))]))


def transfer(config, model, checkpoint, log_dir, dataset):
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=100)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=config['data_load_workers'])

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model, None)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for it, x in enumerate(dataloader):
        out = model(x, True)
        image = Visualizer().visualize_transfer(inp=x, out=out)
        imageio.mimsave(os.path.join(log_dir, str(it).zfill(8) + '.gif'), image)


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

    log_dir = os.path.join(opt.log_dir, opt.config.split('.')[0] + '-' + opt.mode + ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime()))

    model = DDModel(block_expansion=config['block_expansion'],
                    num_channels=config['num_channels'],
                    num_kp=config['num_kp'],
                    kp_gaussian_sigma=config['kp_gaussian_sigma'],
                    deformation_type=config['deformation_type'],
                    kp_extractor_temperature=config['kp_extractor_temperature'])
    model = torch.nn.DataParallel(module=model, device_ids=opt.device_ids)

    data_transform = transforms.Compose([
        VideoToTensor(),
        Normalize(config['spatial_size'])
    ])

    dataset = FramesDataset(root_dir=config['data_dir'], transform=data_transform, offline_kp=config['offline_kp'],
                            offline_flow=config['offline_flow'], image_shape=(config['spatial_size'], config['spatial_size'], 3),
                            is_train=(opt.mode == 'train'), frames_per_sample=config['frames_per_sample'])

    if opt.mode == 'train':
        print ("Start model training...")
        train(config, model, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'test':
        print ("Start model testing...")
        test(config, model, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'transfer':
        print ("Transfering motion...")
        transfer(config, model, opt.checkpoint, log_dir, dataset)

