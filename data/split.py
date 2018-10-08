from sklearn.model_selection import train_test_split
import os
from shutil import move

dataset = 'shapes'
images = os.listdir(dataset)

if not os.path.exists(os.path.join(dataset, 'train')):
    os.makedirs(os.path.join(dataset, 'train'))

if not os.path.exists(os.path.join(dataset, 'test')):
    os.makedirs(os.path.join(dataset, 'test'))

train, test = train_test_split(images, random_state=0, test_size=0.2)


def mv_all_images(images, in_folder, out_folder):
    for img in images:
        move(os.path.join(in_folder, img), os.path.join(out_folder, img))


mv_all_images(train, dataset, os.path.join(dataset, 'train'))
mv_all_images(test, dataset, os.path.join(dataset, 'test'))
