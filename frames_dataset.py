import os
import torch
from skimage import io, img_as_float32
from skimage.color import gray2rgb, rgb2gray
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops
from skimage.draw import circle

import numpy as np
from torch.utils.data import Dataset
import imageio


class VideoToTensor(object):
    def __init__(self, cuda=True):
        self.cuda = cuda
    """Convert video array to Tensor."""
    def __call__(self, sample):
        sample['video_array'] = sample['video_array'].transpose((3, 0, 1, 2))
        # sample['video_array'] = torch.from_numpy(sample['video_array'])
        # if self.cuda:
        #     sample['video_array'] = sample['video_array'].cuda()
        return sample


class NormalizeKP(object):
    def __init__(self, spatial_size, cuda=True):
        self.spatial_size = spatial_size
        self.cuda = cuda

    def __call__(self, sample):
        if 'kp_array' in sample:
            sample['kp_array'] /= self.spatial_size
            # sample['kp_array'] = torch.from_numpy(sample['kp_array'])
            # if self.cuda:
            #     sample['kp_array'] = sample['kp_array'].cuda()
        return sample


class FramesDataset(Dataset):
    """Dataset of videos, represented as image of consequent frames"""
    def __init__(self, root_dir, transform=None, image_shape=(64, 64, 3), is_train=True, random_seed=0,
                 offline_kp=True):
        """
        Args:
            root_dir (string): Path to folder with images
        """
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transform = transform
        self.image_shape = image_shape
        self.offline_kp = offline_kp

        train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

    def __len__(self):
        return len(self.images)

    def compute_kp_for_shapes(self, video_array):
        kp_array = np.empty((video_array.shape[0], 4, 2), dtype='float32')
        for i, image in enumerate(video_array):
            label_map = label(rgb2gray(image) > 0.01)

            for r in regionprops(label_map):
                min_row, min_col, max_row, max_col = r.bbox
                kp_array[i] = [[min_row, min_col],
                               [min_row, max_col],
                               [max_row, max_col],
                               [max_row, min_col]]

        return kp_array

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1, ) + self.image_shape)
        video_array = np.moveaxis(video_array, 1, 2)

        out = {'video_array': video_array}

        if self.offline_kp:
            out['kp_array'] = self.compute_kp_for_shapes(out['video_array'])

        if self.transform:
            out = self.transform(out)

        return out


def draw_video(sample_dict):
    if 'kp_array' in sample_dict:
        video_array = np.copy(sample_dict['video_array'])
        for i in range(len(video_array)):
            for kp in sample_dict['kp_array'][i]:
                rr, cc = circle(kp[0], kp[1], 3, shape=video_array.shape[1:2])
                video_array[i][rr, cc] = (1, 1, 1)
        return video_array
    else:
        return sample_dict['video_array']

if __name__ == "__main__":
    actions_dataset = FramesDataset(root_dir='data/shapes', is_train=True)

    sample = draw_video(actions_dataset[20])

    imageio.mimsave('movie1.gif', sample)
