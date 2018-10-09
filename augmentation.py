"""
Code from https://github.com/hassony2/torch_videovision
"""

import numbers

import random
import numpy as np
import PIL
import scipy

from skimage.transform import resize, rotate
from skimage.util import pad

def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0,0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0,0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')



def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant', anti_aliasing=True) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(img) for img in clip]
 
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)

        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class SelectRandomFrames(object):
    """Select frames from video, to make a batch.
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, reflect_pad_time=True, consequent=True, select_appearance_frame=True):
        self.reflect_pad_time = reflect_pad_time
        self.consequent = consequent
        self.select_appearance_frame = select_appearance_frame
        self.number_of_frames = 100000

    def set_number_of_frames(self, number_of_frames):
        self.number_of_frames = number_of_frames

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images for selection
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: List of number_of_frames images
        """

        if self.reflect_pad_time:
            clip = np.concatenate([clip, clip[::-1]], axis=0)
        frame_count = clip.shape[0]

        num_frames_to_select = self.number_of_frames + self.select_appearance_frame
        if self.consequent:
            first_frame = np.random.choice(max(1, frame_count - num_frames_to_select + 1), size=1)[0]
            selected = clip[first_frame:(first_frame + self.number_of_frames)]
        else:
            selected_index = np.sort(np.random.choice(range(frame_count), replace=False, size=num_frames_to_select))
            selected = clip[selected_index]

        if self.select_appearance_frame:
            index = np.random.choice(frame_count, size=1)[0]
            selected = np.concatenate([selected, clip[index:(index+1)]], axis=0)
            

        return selected


class SplitVideoAppearance(object):
    def __call__(self, video_array):
        video_array = np.array(video_array[:-1], dtype='float32')
        appearance_array = np.array(video_array[-1:], dtype='float32')
        return {'video_array': video_array.transpose((3, 0, 1, 2)),
                'appearance_array': appearance_array.transpose((3, 0, 1, 2))}


class VideoToTensor(object):
    """Convert video array to Tensor."""
    def __call__(self, video_array):
        video_array = np.array(video_array, dtype='float32')
        return {'video_array': video_array.transpose((3, 0, 1, 2))}


class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None,
                 select_frames_param={'reflect_pad_time': True, 'consequent': True}):
        self.transforms = []
        self.select = SelectRandomFrames(**select_frames_param)
        self.transforms.append(self.select)

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

        self.transforms.append(SplitVideoAppearance())

    def set_number_of_frames(self, number_of_frames):
        self.select.set_number_of_frames(number_of_frames)

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

