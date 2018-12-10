import os
import warnings
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from augmentation import AllAugmentationTransform, VideoToTensor


class FramesDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""
    def __init__(self, root_dir, augmentation_params, image_shape=(64, 64, 3), is_train=True,
                 random_seed=0, classes_list=None, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.classes_list = classes_list

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images            
        else:
            self.images = test_images

        if transform is None:
            if is_train:
                self.transform = AllAugmentationTransform(**augmentation_params)
            else:
                self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def set_number_of_frames_per_sample(self, number_of_frames):
        self.transform.set_number_of_frames(number_of_frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        if img_name.lower().endswith('.png') or img_name.lower().endswith('.jpg'):
            image = io.imread(img_name)

            if len(image.shape) == 2 or image.shape[2] == 1:
                image = gray2rgb(image)

            if image.shape[2] == 4:
                image = image[..., :3]

            image = img_as_float32(image)

            video_array = np.moveaxis(image, 1, 0)

            video_array = video_array.reshape((-1, ) + self.image_shape)
            video_array = np.moveaxis(video_array, 1, 2)
        elif img_name.lower().endswith('.gif') or img_name.lower().endswith('.mp4'):
            video = np.array(mimread(img_name))
            if len(video.shape) == 3:
                video = np.array([gray2rgb(frame) for frame in video])
            if video.shape[-1] == 4:
                video = video[..., :3]
            video_array = img_as_float32(video)
        else:
            raise Exception("Unknown file extensions  %s" % img_name)

        out = self.transform(video_array)
        #add names
        out['name'] = os.path.basename(img_name)

        return out


class PairedDataset(Dataset):
    """
    Dataset of pairs for transfer.
    """
    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        classes_list = self.initial_dataset.classes_list

        np.random.seed(seed)
 
        if classes_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx,:ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            images = self.initial_dataset.images
            name_to_index = {name:index for index, name in enumerate(images)}
            classes = pd.read_csv(classes_list)
            classes = classes[np.logical_and(classes['appearance'].isin(images), classes['video'].isin(images))]
            
            number_of_pairs = min(classes.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[classes['video'].iloc[ind]], name_to_index[classes['appearance'].iloc[ind]]))

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
