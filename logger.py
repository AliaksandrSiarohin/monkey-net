import numpy as np
import torch
import imageio

import os
from modules.kp_extractor import kp2gaussian
from skimage.draw import circle

import matplotlib.pyplot as plt



class Logger:
    def __init__(self, generator, log_dir, discriminator=None, optimizer_generator=None,
                 optimizer_discriminator=None, log_file_name='log.txt', log_freq=100, cpk_freq=1000, fill_counter=8):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.loss_list = []
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.log_freq = log_freq
        self.cpk_freq = cpk_freq
        self.fill_counter = fill_counter

        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator

        self.it = 0

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.it).zfill(self.fill_counter) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp):
        out = self.generator(inp)
        image = Visualizer().visualize_reconstruction(inp, out)
        imageio.mimsave(os.path.join(self.log_dir, "%s-rec.gif" % str(self.it).zfill(self.fill_counter)), image)

    def save_cpk(self):
        d = {"generator": self.generator.state_dict(),
             "optimizer_generator": self.optimizer_generator.state_dict(),
             "discriminator": self.discriminator.state_dict(),
             "optimizer_discriminator": self.optimizer_discriminator.state_dict(),
             "iter": self.it}
        torch.save(d, os.path.join(self.log_dir, '%s-checkpoint.pth.tar' % str(self.it).zfill(self.fill_counter)))

    @staticmethod
    def load_cpk(checkpoint_path, generator, discriminator=None,
                 optimizer_generator=None, optimizer_discriminator=None):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        return checkpoint['iter']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_cpk()
        self.log_file.close()

    def save_values(self, names, values):
        self.names = names
        self.loss_list.append(values)

    def log(self, it, inp):
        self.it = it
        if it % self.log_freq == 0:
            self.log_scores(self.names)
            self.visualize_rec(inp)
            #self.visualize_transfer(inp)

        if it % self.cpk_freq == 0:
            self.save_cpk()


class Visualizer:
    def __init__(self, kp_size=2, draw_border=True, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_video_with_kp(self, video, kp_array):
        video_array = np.copy(video)
        spatial_size = np.array(video_array.shape[2:0:-1])[np.newaxis, np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[1]
        for i in range(len(video_array)):
            for kp_ind, kp in enumerate(kp_array[i]):
                rr, cc = circle(kp[1], kp[0], self.kp_size, shape=video_array.shape[1:3])
                video_array[i][rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return video_array

    def create_video_column_with_kp(self, video, kp):
        video_array = np.array([self.draw_video_with_kp(v, k) for v, k in zip(video, kp)])
        return self.create_video_column(video_array)

    def create_video_column(self, videos):
        if self.draw_border:
            videos = np.copy(videos)
            videos[:, :, [0, -1]] = (1, 1, 1)
            videos[:, :, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(videos), axis=1)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_video_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_video_column(arg))
        return np.concatenate(out, axis=2)

    def visualize_transfer(self, inp, out):
        out_video_batch = out['video_prediction'].data.cpu().numpy()
        appearance_deformed_batch = out['video_deformed'].data.cpu().numpy()
        motion_video_batch = inp['first_video_array'].data.cpu().numpy()
        appearance_video_batch = inp['second_video_array'][:, :, 0:1].repeat(1, 1, out_video_batch.shape[2], 1, 1).data.cpu().numpy()

        kp_video = out['kp_video']['mean'].data.cpu().numpy()
        kp_appearance = out['kp_appearance']['mean'].data.repeat(1, out_video_batch.shape[2], 1, 1).data.cpu().numpy()

        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])
        motion_video_batch = np.transpose(motion_video_batch, [0, 2, 3, 4, 1])
        appearance_video_batch = np.transpose(appearance_video_batch, [0, 2, 3, 4, 1])
        appearance_deformed_batch = np.transpose(appearance_deformed_batch, [0, 2, 3, 4, 1])

        image = self.create_image_grid((appearance_video_batch, kp_appearance), (motion_video_batch, kp_video),
                                       out_video_batch, appearance_deformed_batch)
        image = (255 * image).astype(np.uint8)
        return image

    def visualize_reconstruction(self, inp, out):
        out_video_batch = out['video_prediction'].data.cpu().numpy()
        gt_video_batch = inp['video_array'].data.cpu().numpy()
        appearance_deformed_batch = out['video_deformed'].data.cpu().numpy()
        kp_array = out['kp_video']['mean'].data.cpu().numpy()

        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])
        gt_video_batch = np.transpose(gt_video_batch, [0, 2, 3, 4, 1])
        appearance_deformed_batch = np.transpose(appearance_deformed_batch, [0, 2, 3, 4, 1])

        diff_batch = gt_video_batch * 0.5 + appearance_deformed_batch * 0.5

        image = self.create_image_grid((gt_video_batch, kp_array), out_video_batch,
                                       appearance_deformed_batch, diff_batch)
        image = (255 * image).astype(np.uint8)
        return image
