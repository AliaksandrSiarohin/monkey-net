import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb, rgb2gray
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops

import numpy as np
from torch.utils.data import Dataset
import imageio

from skimage.transform import estimate_transform, warp_coords


class VideoToTensor(object):
    def __init__(self, cuda=True):
        self.cuda = cuda
    """Convert video array to Tensor."""
    def __call__(self, sample):
        sample['video_array'] = sample['video_array'].transpose((3, 0, 1, 2))
        return sample


class Normalize(object):
    def __init__(self, spatial_size):
        self.spatial_size = np.array(spatial_size)

    def __call__(self, sample):
        if 'kp_array' in sample:
            sample['kp_array'] /= (self.spatial_size[np.newaxis, np.newaxis] - 1)
            sample['kp_array'] *= 2
            sample['kp_array'] -= 1
        if 'flow_array' in sample:
            sample['flow_array'] /= (self.spatial_size[np.newaxis, np.newaxis, np.newaxis] - 1)
            sample['flow_array'] *= 2
        return sample


class FramesDataset(Dataset):
    """Dataset of videos, represented as image of consequent frames"""
    def __init__(self, root_dir, transform=None, image_shape=(64, 64, 3), is_train=True, random_seed=0,
                 offline_kp=True, offline_flow=True, frames_per_sample=1000000):
        """
        Args:
            root_dir (string): Path to folder with images
        """
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transform = transform
        self.image_shape = tuple(image_shape)
        self.offline_kp = offline_kp
        self.offline_flow = offline_flow
        self.frames_per_sample = frames_per_sample

        train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

    def __len__(self):
        return len(self.images)

    def set_number_of_frames_per_sample(self, number_of_frames):
        self.frames_per_sample = number_of_frames

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

    def compute_optical_flow_for_shapes(self, video_array):
        kp_array = self.compute_kp_for_shapes(video_array)

        kp_array = kp_array[:, :, np.newaxis, np.newaxis, :]

        flow_array = np.zeros((kp_array.shape[0], video_array.shape[1], video_array.shape[2], 2), dtype=np.float32)
        for i in range(kp_array.shape[0]):
            flow_array[i] = kp_array[0, 0] - kp_array[i, 0]

        return flow_array

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

        frame_count = video_array.shape[0]
        first_frame = np.random.choice(frame_count - self.frames_per_sample + 1, size=1)[0]

        video_array = video_array[first_frame:(first_frame + self.frames_per_sample)]

        out = {'video_array': video_array}

        if self.offline_kp:
            out['kp_array'] = self.compute_kp_for_shapes(out['video_array'])

        if self.offline_flow:
            out['flow_array'] = self.compute_optical_flow_for_shapes(video_array)

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
    actions_dataset = FramesDataset(root_dir='data/actions', is_train=True)

    video = actions_dataset[20]['video_array']
    from skimage.io import imsave

    imsave('1.png', video[0])
    imsave('2.png', video[1])

    #imageio.mimsave('movie.gif', video)
