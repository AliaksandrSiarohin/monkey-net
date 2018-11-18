from tqdm import trange

import torch
from torch.utils.data import DataLoader

from logger import Logger
from modules.losses import generator_loss, discriminator_loss, generator_loss_names, discriminator_loss_names

import numpy as np
from torch.autograd import Variable
import warnings
import gc
from sync_batchnorm import DataParallelWithCallback



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


class GeneratorFullModel(torch.nn.Module):
    def __init__(self, kp_extractor, generator, discriminator, config):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config

    def forward(self, x):
        kp_joined = self.kp_extractor(torch.cat([x['appearance_array'], x['video_array']], dim=2))
        generated = self.generator(x['appearance_array'], **split_kp(kp_joined, self.config['model_params']['detach_kp_generator']))
        video_prediction = generated['video_prediction']
        video_deformed = generated['video_deformed']

        kp_dict = split_kp(kp_joined, False)
        discriminator_maps_generated = self.discriminator(video_prediction, **kp_dict)
        discriminator_maps_real = self.discriminator(x['video_array'], **kp_dict)
        generated.update(kp_dict)

        if self.config['loss_weights']['reconstruction_deformed'] is not None:
            if np.abs(self.config['loss_weights']['reconstruction_deformed'][1:]).sum() == 0:
                        #Only reconstruction of deformed_video
                discriminator_maps_deformed = [video_deformed] + [None] * (len(discriminator_maps_real) - 1)
            else:
                discriminator_maps_deformed = self.discriminator(video_deformed, **kp_dict)
        else:
            discriminator_maps_deformed = None
        loss = generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                                         discriminator_maps_deformed=discriminator_maps_deformed,
                                                         discriminator_maps_real=discriminator_maps_real,
                                                         loss_weights=self.config['loss_weights'])
        
#        self.gen_values = gen_values
#        self.gen_names = gen_names          
        return  tuple(loss) + (generated, kp_joined)
 

class DiscriminatorFullModel(torch.nn.Module):
    def __init__(self, kp_extractor, generator, discriminator, config):
        super(DiscriminatorFullModel, self).__init__() 
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config

    def forward(self, x, kp_joined, generated):
        kp_dict = split_kp(kp_joined, self.config['model_params']['detach_kp_discriminator'])
        discriminator_maps_generated = self.discriminator(generated['video_prediction'].detach(), **kp_dict)
        discriminator_maps_real = self.discriminator(x['video_array'], **kp_dict)
        loss = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                  discriminator_maps_real=discriminator_maps_real,
                                  loss_weights=self.config['loss_weights'])
#       self.disc_values = disc_value
#       self.disc_names = disc_names
        return loss


def train(config, generator, discriminator, kp_extractor, checkpoint, log_dir, dataset, device_ids):
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

    generator_full = GeneratorFullModel(kp_extractor, generator, discriminator, config) 
    discriminator_full = DiscriminatorFullModel(kp_extractor, generator, discriminator, config) 
 
    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full_par = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(generator=generator, discriminator=discriminator, kp_extractor=kp_extractor, optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator, optimizer_kp_extractor=optimizer_kp_extractor,
                log_dir=log_dir, **config['log_params']) as logger:
        for epoch in trange(start_epoch, epochs_milestones[-1]):
            for x in dataloader:
                out = generator_full_par(x)
                loss_values = out[:-2]
                generated = out[-2]
                kp_joined = out[-1]
                loss_values = [val.mean() for val in loss_values]
                generator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                loss = sum(loss_values) 
                loss.backward(retain_graph=not config['model_params']['detach_kp_discriminator'])                      
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()
                if config['model_params']['detach_kp_discriminator']:
                    optimizer_kp_extractor.step()
                    optimizer_kp_extractor.zero_grad()               

                loss_values = discriminator_full_par(x, kp_joined, generated)

                loss_values = [val.mean() for val in loss_values]
                discriminator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                loss = sum(loss_values) 
                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if not config['model_params']['detach_kp_discriminator']:
                    optimizer_kp_extractor.step()
                    optimizer_kp_extractor.zero_grad()
 
                logger.log_iter(it, names=generator_loss_names(config['loss_weights']) + discriminator_loss_names(),
                                values=generator_loss_values + discriminator_loss_values, inp=x, out=generated)
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
