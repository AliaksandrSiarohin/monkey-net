#!/usr/bin/env python
'''
===============================================================================
BG removal for .gif images.

This sample shows interactive removal of bg.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

Key '0' - To fill with white the area.
Key '1' - To replace with white a color.
Key '2' - To replace with white a color in all frames.
Key '3' - Not Implimented

Key 'n' - To next image
Key 's' - To skip
Key 'r' - To reset the setup
Key 'f' - To fill with random color
Key 'd' - dilation
Key 'i' - invert colors
Key 'p' - denoising
Key 'l' - pause
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

from imageio import mimsave
import os

from skimage.measure import label
from skimage.segmentation import quickshift
from  skimage.morphology import binary_dilation, binary_erosion

import scipy

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import img_as_ubyte

from skimage.filters.rank import median
from skimage.morphology import disk

FILL_AREA = 1
FILL_COLOR = 2
FILL_ALL_COLOR = 3
FILL_CONNECTED = 4

drawing = False
value = FILL_COLOR
thickness_area = 7
thickness_color = 5

mask = None

image_shape = (256, 256)



def get_files_by_file_size(filepaths, dir, reverse=False):
    """ Return list of file paths  sorted by file size """

    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(os.path.join(dir, filepaths[i])))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]

    return filepaths


def onmouse(event, x, y, flags, param):
    global drawing, mask
    thickness = thickness_area if value == FILL_AREA else thickness_color
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(mask, (x, y), thickness, True, -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(mask, (x, y), thickness, True, -1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(mask, (x, y), thickness, True, -1)


def convert_gif_to_frames(gif):
    # Initialize the frame number and create empty frame list
    frame_num = 0
    frame_list = []

    # Loop until there are frames left
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not
            okay, frame = gif.read()

            if not okay:
                break
            # Append to empty frame list
            frame = cv.resize(frame, image_shape, interpolation=cv.INTER_NEAREST)
            frame_list.append(frame)
            # Break if there are no other frames to read

            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

    return frame_list


def color_as_num(val):
    val = val.astype(np.uint64)
    return val[..., 0] + 256 * val[..., 1] + (256 * 256) * val[..., 2]


def process(video, filename, outdir, skipdir):
    global mask, value, drawing
    video2 = video.copy()
    current_it = 0

    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', video.shape[2] + 10, 90)

    video_as_num = color_as_num(video)
    original_fillmask = video_as_num != (256 ** 3 - 1)

    paused = False

    while (1):
        current_frame = current_it // 25
        cv.imshow('input', video[current_frame % video.shape[0]])
        k = cv.waitKey(1)
        if not paused:
            current_it += 1

        # key bindings
        if k == 27:  # esc to exit
            break
        elif k == ord('0'):  # BG drawing
            print(" Mark region to fill with left mouse button \n")
            value = FILL_AREA
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('1'):  # FG drawing
            print("Mark colors to fill with left mouse button \n")
            value = FILL_COLOR
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('2'):  # PR_BG drawing
            print("Mark area to fill (in  all frames) with left mouse button \n")
            value = FILL_ALL_COLOR
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('3'):  # PR_BG drawing
            print("Mark connected area to fill with left mouse button \n")
            value = FILL_CONNECTED
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('f'):  # fill with random background
            color = np.random.randint(255, size=3)
            video_as_num = color_as_num(video)
            fillmask = video_as_num != (256 ** 3 - 1)
            for i in range(len(video)):
                video[i, scipy.ndimage.morphology.binary_fill_holes(fillmask[i])] = color
            video[original_fillmask] = (0, 0, 0)
        elif k == ord('d'): #dilation of image
            video_as_num = color_as_num(video)
            fillmask = video_as_num != (256 ** 3 - 1)
            for i in range(len(video)):
                video[i, binary_dilation(fillmask[i])] = (0, 0, 0)
        elif k == ord('e'): #erosion of image
            video_as_num = color_as_num(video)
            fillmask = video_as_num != (256 ** 3 - 1)
            for i in range(len(video)):
                video[i, np.logical_not(binary_erosion(fillmask[i]))] = (255, 255, 255)
        elif k == ord('i'): #inversion of colors
            video = 255 - video
        elif k == ord('p'): #image denoising
            video = np.array([img_as_ubyte(np.concatenate([median(frame[..., i], disk(1))[..., np.newaxis] for i in range(3)], axis=-1)) for frame in video])
        elif k == ord('l'): #pause animation
            paused = not paused
        elif k == ord('n'):  # save image
            mimsave(os.path.join(outdir, filename), video[..., ::-1])
            break
        elif k == ord('s'):  # skip image
            mimsave(os.path.join(skipdir, filename), video2[..., ::-1])
            break
        elif k == ord('r'):  # reset everything
            print("resetting \n")
            drawing = False
            video = video2.copy()
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)

        if mask.sum() == 0:
            continue

        if value == FILL_AREA:
            video[:, mask.astype(bool)] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_COLOR:
            colors = video[current_frame % video.shape[0]][mask.astype(bool)]
            colors = color_as_num(val=colors).reshape((-1, ))
            colors = np.unique(colors)
            video_as_num = color_as_num(video)
            for color in colors:
                video[video_as_num == color] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_ALL_COLOR:
            colors = video[:, mask.astype(bool)]
            colors = color_as_num(val=colors).reshape((-1, ))
            colors = np.unique(colors)
            video_as_num = color_as_num(video)
            for color in colors:
                video[video_as_num == color] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_CONNECTED:
            color = np.random.randint(255, size=3)
            video_as_num = color_as_num(video)
            fillmask = video_as_num != (256 ** 3 - 1)
            for i in range(len(video)):
                labels = label(fillmask[i])
                index = labels[mask]
                video[i, labels == np.unique(index)] = color
            video[original_fillmask] = (0, 0, 0)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)


    cv.destroyAllWindows()


if __name__ == '__main__':
    # print documentation
    print(__doc__)

    # Loading images
    if len(sys.argv) == 4:
        inputdir = sys.argv[1]
        outdir = sys.argv[2]
        skipdir = sys.argv[3]
    else:
        print("Correct Usage: python grabcut.py <inputdir> <outputdir> <skipdir> \n")
        exit()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(skipdir):
        os.makedirs(skipdir)

    input_list = set(os.listdir(inputdir))
    output_list = set(os.listdir(outdir))
    skip_list = set(os.listdir(skipdir))

    videos = sorted(list(input_list - output_list - skip_list))
    videos = get_files_by_file_size(videos, inputdir)

    for filename in videos:
        print (filename)
        video = []
        video = convert_gif_to_frames(cv.VideoCapture(os.path.join(inputdir,filename)))
        video = np.array(video)
        mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        process(video, filename, outdir, skipdir)
