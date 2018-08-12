import numpy as np
from skimage.draw import circle
import torch
import imageio

import os


class Logger:
    def __init__(self, model, log_dir, optimizer=None, log_file_name='log.txt', log_freq=100, cpk_freq=1000, fill_counter=8):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.loss_list = []
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.log_freq = log_freq
        self.cpk_freq = cpk_freq
        self.fill_counter = fill_counter

        self.model = model
        self.optimizer = optimizer

        self.it = 0

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.it).zfill(self.fill_counter) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize(self, inp, out):
        out_video_batch = out['video_prediction'].detach().cpu().numpy()
        gt_video_batch = inp['video_array'].detach().cpu().numpy()
        appearance_deformed_batch = out['video_deformed'].detach().cpu().numpy()

        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])
        gt_video_batch = np.transpose(gt_video_batch, [0, 2, 3, 4, 1])
        appearance_deformed_batch = np.transpose(appearance_deformed_batch, [0, 2, 3, 4, 1])

        diff_batch = gt_video_batch * 0.5 + appearance_deformed_batch * 0.5

        image = Visualizer(kp_size=2).create_image_grid(gt_video_batch, out_video_batch,
                                                        appearance_deformed_batch, diff_batch)
        image = (255 * image).astype(np.uint8)

        imageio.mimsave(os.path.join(self.log_dir, "sample%s.gif" % str(self.it).zfill(self.fill_counter)), image)

    def save_cpk(self):
        d = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "iter": self.it}
        torch.save(d, os.path.join(self.log_dir, 'checkpoint%s.pth.tar' % str(self.it).zfill(self.fill_counter)))

    @staticmethod
    def load_cpk(checkpoint_path, model, optimizer = None):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['iter']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_cpk()
        self.log_file.close()

    def log(self, it, loss_list, inp, out):
        names, values = list(zip(*loss_list))
        self.it = it

        self.loss_list.append(values)

        if it % self.log_freq == 0:
            self.log_scores(names)
            self.visualize(inp, out)

        if it % self.cpk_freq == 0:
            self.save_cpk()







class Visualizer:
    def __init__(self, kp_size=2):
        self.kp_size = kp_size

    def draw_video_with_kp(self, video, kp_array):
        print (video.shape, kp_array.shape)
        video_array = np.copy(video)
        for i in range(len(video_array)):
            for kp in kp_array[i]:
                rr, cc = circle(kp[0], kp[1], self.kp_size, shape=video_array.shape[1:2])
                video_array[i][rr, cc] = (1, 1, 1)
        return video_array

    def create_video_column_with_kp(self, video, kp):
        video_array = np.array([self.draw_video_with_kp(v, k) for v, k in zip(video, kp)])
        return self.create_video_column(video_array)

    def create_video_column(self, videos):
        return np.concatenate(list(videos), axis=1)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_video_column(arg))
        return np.concatenate(out, axis=2)

