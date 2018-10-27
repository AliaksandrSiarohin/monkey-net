from skimage.io import imread, imsave
from skimage.transform import resize
import os
from tqdm import tqdm
import numpy as np

in_dir = 'unzippedIntervalFaces/data/%s/1.6/'
img_size = (256, 256)
out_dir = 'vox'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for partition in ['train', 'test']:
    par_dir = os.path.join(out_dir, partition)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    celebs = open(partition + '_vox1.txt').read().splitlines()
    for celeb in tqdm(celebs):
        celeb_dir = in_dir % celeb
        for video_dir in os.listdir(celeb_dir):
            for part_dir in os.listdir(os.path.join(celeb_dir, video_dir)):
                result_name = celeb + "-" + video_dir + "-" + part_dir + ".jpg"
                part_dir = os.path.join(celeb_dir, video_dir, part_dir) 
                images = [resize(imread(os.path.join(part_dir, img_name)), img_size)  for img_name in sorted(os.listdir(part_dir))]
                if len(images) > 100 or len(images) < 4:
                    print ("Warning sequence of len - %s" % len(images))
                result = np.concatenate(images, axis=1)
                imsave(os.path.join(par_dir, result_name), result)

