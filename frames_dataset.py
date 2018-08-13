import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb, rgb2gray
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops

import numpy as np
from torch.utils.data import Dataset
import imageio


class VideoToTensor(object):
    def __init__(self, cuda=True):
        self.cuda = cuda
    """Convert video array to Tensor."""
    def __call__(self, sample):
        sample['video_array'] = sample['video_array'].transpose((3, 0, 1, 2))
        return sample


class NormalizeKP(object):
    def __init__(self, spatial_size, cuda=True):
        self.spatial_size = spatial_size
        self.cuda = cuda

    def __call__(self, sample):
        if 'kp_array' in sample:
            sample['kp_array'] /= (self.spatial_size - 1)
            sample['kp_array'] *= 2
            sample['kp_array'] -= 1
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
                kp_array[i] = [[min_col, min_row],
                               [max_col, min_row],
                               [max_col, max_row],
                               [min_col, max_row]]

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


class PairedDataset(Dataset):
    """
    Dataset of pairs.
    """
    def __init__(self, initial_dataset, number_of_pairs, seed = 0):
        self.initial_dataset = initial_dataset

        max_idx = min(number_of_pairs, len(initial_dataset))
        nx, ny = max_idx, max_idx
        xy = np.mgrid[:nx,:ny].reshape(2, -1).T

        number_of_pairs = min(xy.shape[0], number_of_pairs)

        np.random.seed(seed)

        self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]

        first = {'first_' + key: value for key, value in first.items()}
        second = {'second_' + key: value for key, value in second.items()}

        return {**first, **second}

if __name__ == "__main__":
    from logger import Visualizer
    actions_dataset = FramesDataset(root_dir='data/shapes', is_train=True)

    video = np.array([actions_dataset[19]['video_array'], actions_dataset[20]['video_array']])
    kp_array = np.array([actions_dataset[19]['kp_array'], actions_dataset[20]['kp_array']])

    sample = Visualizer(kp_size=2).create_video_column(video)

    imageio.mimsave('movie1.gif', sample)
