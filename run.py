from argparse import ArgumentParser
import os
import yaml
from time import gmtime, strftime
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import FramesDataset, PairedDataset
from logger import Logger, Visualizer
from modules.dd_model import DDModel
from modules.discriminator import Discriminator
from modules.losses import generator_loss, reconstruction_loss, discriminator_loss

import numpy as np
import imageio
from torch.autograd import Variable

import warnings


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(config, generator, discriminator, checkpoint, log_dir, dataset):
    start_iter = 0
    optimizer_generator = torch.optim.Adam(generator.parameters(), betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.999))

    if checkpoint is not None:
        start_iter = Logger.load_cpk(checkpoint, generator, discriminator, optimizer_generator, optimizer_discriminator)

    epochs_milestones = np.cumsum(config['schedule_params']['num_epochs'])

    schedule_iter = np.searchsorted(epochs_milestones, start_iter)
    dataloader = DataLoader(dataset, batch_size=config['schedule_params']['batch_size'][schedule_iter],
                            shuffle=True, num_workers=4)
    set_optimizer_lr(optimizer_generator, config['schedule_params']['lr_generator'][schedule_iter])
    set_optimizer_lr(optimizer_discriminator, config['schedule_params']['lr_discriminator'][schedule_iter])

    dataset.set_number_of_frames_per_sample(config['schedule_params']['frames_per_sample'][schedule_iter])

    with Logger(generator=generator, discriminator=discriminator, optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator, log_dir=log_dir, **config['log_params']) as logger:
        for it in trange(start_iter, epochs_milestones[-1]):
            for i, x in enumerate(dataloader):
                x = {k: Variable(x[k], requires_grad=True) for k,v in x.items()}
                generated = generator(x)

                video_prediction = generated['video_prediction']
                video_deformed = generated['video_deformed']

                discriminator_maps_generated = discriminator(video_prediction)
                discriminator_maps_real = discriminator(x['video_array'])

                if config['loss_weights']['reconstruction_deformed'] is not None:
                    discriminator_maps_deformed = discriminator(video_deformed)
                else:
                    discriminator_maps_deformed = None

                loss, gen_loss_names, gen_loss_values = generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                                                       discriminator_maps_deformed=discriminator_maps_deformed,
                                                                       discriminator_maps_real=discriminator_maps_real,
                                                                       loss_weights=config['loss_weights'],
                                                                       deformation=generated['deformation'],
                                                                       kp_video=generated['kp_video'])
                loss.backward()

                if torch.isnan(x['video_array'].grad).byte().any():
                    warnings.warn("Nan in gradient", Warning)
                    optimizer_generator.zero_grad()
                    optimizer_discriminator.zero_grad()
                    continue
                else:
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_discriminator.zero_grad()

                discriminator_maps_generated = discriminator(video_prediction.detach())
                discriminator_maps_real = discriminator(x['video_array'])
                loss, disc_loss_names, disc_loss_values = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                                                             discriminator_maps_real=discriminator_maps_real,
                                                                             loss_weights=config['loss_weights'])

                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

                logger.save_values(gen_loss_names + disc_loss_names, gen_loss_values + disc_loss_values)
                logger.log(i, inp=x)

            if it in epochs_milestones:
                schedule_iter = np.searchsorted(epochs_milestones, it, side='right')
                lr_generator = config['schedule_params']['lr_generator'][schedule_iter]
                lr_discriminator = config['schedule_params']['lr_discriminator'][schedule_iter]
                bs = config['schedule_params']['batch_size'][schedule_iter]
                frames_per_sample = config['schedule_params']['frames_per_sample'][schedule_iter]
                print("Schedule step: lr - %s, bs - %s, frames_per_sample - %s" % ((lr_generator, lr_discriminator), bs, frames_per_sample))
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)
                set_optimizer_lr(optimizer_generator, lr_generator)
                set_optimizer_lr(optimizer_discriminator, lr_discriminator)
                dataset.set_number_of_frames_per_sample(frames_per_sample)

            logger.log(it, inp=x)


def test(config, generator, checkpoint, log_dir, dataset):
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    loss_list = []
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            out = generator(x)
            image = Visualizer().visualize_reconstruction(x, out)
            imageio.mimsave(os.path.join(log_dir, str(it).zfill(8) + '.gif'), image)
            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video_array'].cpu(),
                                       config['loss_weights']['reconstruction'][0])
            loss_list.append(loss.data.cpu().numpy())

    print ("Reconstruction loss: %s" % np.mean(loss_list))


def transfer(config, generator, checkpoint, log_dir, dataset):
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=100)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = generator.module
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value.cuda() for key,value in x.items()}
            out = model.transfer(x)
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

    generator = DDModel(**config['generator_params'])
    generator = torch.nn.DataParallel(module=generator, device_ids=opt.device_ids)

    discriminator = Discriminator(**config['discriminator_params'])
    discriminator = torch.nn.DataParallel(module=discriminator, device_ids=opt.device_ids)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print ("Start model training...")
        train(config, generator, discriminator, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'test':
        print ("Start model testing...")
        test(config, generator, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'transfer':
        print ("Transfering motion...")
        transfer(config, generator, opt.checkpoint, log_dir, dataset)

