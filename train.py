from tqdm import trange

import torch
from torch.utils.data import DataLoader

from logger import Logger
from modules.losses import generator_loss, discriminator_loss

import numpy as np
from torch.autograd import Variable
import warnings


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return {'kp_video': kp_video, 'kp_appearance': kp_appearance}


def train(config, generator, discriminator, kp_extractor, checkpoint, log_dir, dataset):
    start_epoch = 0
    it = 0

    optimizer_generator = torch.optim.Adam(generator.parameters(), betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.999))
    optimizer_kp_extractor = torch.optim.Adam(kp_extractor.parameters(), betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch, it = Logger.load_cpk(checkpoint, generator, discriminator, kp_extractor,
                                          optimizer_generator, optimizer_discriminator, optimizer_kp_extractor=None)

    epochs_milestones = np.cumsum(config['schedule_params']['num_epochs'])

    schedule_iter = np.searchsorted(epochs_milestones, start_epoch)
    dataloader = DataLoader(dataset, batch_size=config['schedule_params']['batch_size'][schedule_iter],
                            shuffle=True, num_workers=4, drop_last=True)
    set_optimizer_lr(optimizer_generator, config['schedule_params']['lr_generator'][schedule_iter])
    set_optimizer_lr(optimizer_discriminator, config['schedule_params']['lr_discriminator'][schedule_iter])
    set_optimizer_lr(optimizer_kp_extractor, config['schedule_params']['lr_kp_extractor'][schedule_iter])

    dataset.set_number_of_frames_per_sample(config['schedule_params']['frames_per_sample'][schedule_iter])
    with Logger(generator=generator, discriminator=discriminator, kp_extractor=kp_extractor, optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator, optimizer_kp_extractor=optimizer_kp_extractor,
                log_dir=log_dir, **config['log_params']) as logger:
        for epoch in trange(start_epoch, epochs_milestones[-1]):
            for x in dataloader:
                x = {key: value.cuda().requires_grad() for key,value in x.items()}

                #Concatenate appearance and video to properly work with batchnorm
                kp_joined = kp_extractor(torch.cat([x['appearance_array'], x['video_array']], dim=2))

                generated = generator(x['appearance_array'][:, :, :1],
                                      **split_kp(kp_joined, config['model_params']['detach_kp_generator']))

                video_prediction = generated['video_prediction']
                video_deformed = generated['video_deformed']

                kp_dict = split_kp(kp_joined, False)
                discriminator_maps_generated = discriminator(video_prediction, **kp_dict)
                discriminator_maps_real = discriminator(x['video_array'], **kp_dict)
                generated.update(kp_dict)

                if config['loss_weights']['reconstruction_deformed'] is not None:
                    if np.abs(config['loss_weights']['reconstruction_deformed'][1:]).sum() == 0:
                        #Only reconstruction of deformed_video
                        discriminator_maps_deformed = [video_deformed] + [None] * (len(discriminator_maps_real) - 1)
                    else:
                        discriminator_maps_deformed = discriminator(video_deformed, **kp_dict)
                else:
                    discriminator_maps_deformed = None

                loss, gen_loss_names, gen_loss_values = generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                                                       discriminator_maps_deformed=discriminator_maps_deformed,
                                                                       discriminator_maps_real=discriminator_maps_real,
                                                                       loss_weights=config['loss_weights'])
                
                if torch.isnan(x['video_array'].grad).byte().any():
                    warnings.warn("Nan in gradient", Warning)
                    optimizer_generator.zero_grad()
                    optimizer_discriminator.zero_grad()
                    optimizer_kp_extractor.zero_grad()
                 
                loss.backward(retain_graph=not config['model_params']['detach_kp_discriminator'])

                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()

                kp_dict = split_kp(kp_joined, config['model_params']['detach_kp_discriminator'])
                discriminator_maps_generated = discriminator(video_prediction.detach(), **kp_dict)
                discriminator_maps_real = discriminator(x['video_array'], **kp_dict)
                loss, disc_loss_names, disc_loss_values = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                                                             discriminator_maps_real=discriminator_maps_real,
                                                                             loss_weights=config['loss_weights'])

                loss.backward()
                optimizer_kp_extractor.step()
                optimizer_kp_extractor.zero_grad()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

                logger.log_iter(it, names=gen_loss_names + disc_loss_names,
                                values=gen_loss_values + disc_loss_values, inp=x, out=generated)
                it += 1

            if epoch in epochs_milestones:
                schedule_iter = np.searchsorted(epochs_milestones, epoch, side='right')
                lr_generator = config['schedule_params']['lr_generator'][schedule_iter]
                lr_discriminator = config['schedule_params']['lr_discriminator'][schedule_iter]
                lr_kp_extractor = config['schedule_params']['lr_kp_extractor'][schedule_iter]
 
                bs = config['schedule_params']['batch_size'][schedule_iter]
                frames_per_sample = config['schedule_params']['frames_per_sample'][schedule_iter]
                print("Schedule step: lr - %s, bs - %s, frames_per_sample - %s" % ((lr_generator, lr_discriminator, lr_kp_extractor), bs, frames_per_sample))
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)
                set_optimizer_lr(optimizer_generator, lr_generator)
                set_optimizer_lr(optimizer_discriminator, lr_discriminator)
                set_optimizer_lr(optimizer_kp_extractor, lr_kp_extractor)
                dataset.set_number_of_frames_per_sample(frames_per_sample)

            logger.log_epoch(epoch)
