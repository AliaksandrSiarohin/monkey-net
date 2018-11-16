import matplotlib
matplotlib.use('Agg')

import torch
import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.dd_model import DDModel
from modules.discriminator import Discriminator
from modules.kp_extractor import KPExtractor

from train import train
from reconstruction import reconstruction
from transfer import transfer
from prediction import prediction

from sync_batchnorm import  DataParallelWithCallback

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "transfer", "prediction"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)
        assert [len(v) == len(config['schedule_params']['num_epochs']) for k, v in config['schedule_params'].items()]

        blocks_discriminator = config['model_params']['discriminator_params']['num_blocks'] +  \
                               (config['model_params']['discriminator_params']['non_local_index'] is not None)
        assert len(config['loss_weights']['reconstruction']) == blocks_discriminator + 1

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0] + ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime()))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    generator = DDModel(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    generator.to(opt.device_ids[0])
    generator = DataParallelWithCallback(module=generator, device_ids=opt.device_ids)
    print(generator)

    discriminator = Discriminator(**config['model_params']['discriminator_params'], **config['model_params']['common_params'])
    discriminator.to(opt.device_ids[0])
    discriminator = DataParallelWithCallback(module=discriminator, device_ids=opt.device_ids)
    print(discriminator)

    kp_extractor = KPExtractor(**config['model_params']['kp_extractor_params'], **config['model_params']['common_params'])
    kp_extractor.to(opt.device_ids[0])
    kp_extractor = DataParallelWithCallback(module=kp_extractor, device_ids=opt.device_ids)
    print(kp_extractor)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_extractor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_extractor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'transfer':
        print("Transfer...")
        transfer(config, generator, kp_extractor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == "prediction":
        print("Prediction...")
        prediction(config, generator, kp_extractor, opt.checkpoint, log_dir)

