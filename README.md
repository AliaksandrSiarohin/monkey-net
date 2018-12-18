# Animating Arbitrary Objects via Deep Motion Transfer

This repository contains the source code for paper [Animating Arbitrary Objects via Deep Motion Transfer]().

![Screenshot](sup-mat/teaser.gif)

### Requirements
You will need ```python3```.
```
pip install -r requirements.txt
```

### General notes

There several config (```dataset_name.yaml```) files one for each dataset. Check ```actions.yaml``` for description of each individual parameter.

### Demo Transfer

In order to run a demo, use the following command:
```
python --config moving-gif.yaml --driving_video sup-mat/driving_video.gif --source_image sup-mat/source_image.gif --checkpoint path/to/checkpoint
```
The result will be stored in ```demo.gif```.

### Training

In order to train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0 python run.py dataset_name.yaml
```
This will create a folder in log directory (each run create new directory).
Checkpoints will be saved in this folder.
You can check the loss values during training in ```log.txt```.
You can also check train reconstructions in ```train-vis``` subfolder.

### Reconstruction

In order to check the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint
```
You will need to specify a path to checkpoint,
the ```reconstruction``` subfolder will be created in the same folder as a checkpoint.
You can find generated video there and in ```png``` subfolder loss-less verstion in '.png' format.

### Transfer

In order to perform a transfer run:
```
CUDA_VISIBLE_DEVICES=0 python run.py dataset_name.yaml --mode transfer --checkpoint path/to/checkpoint
```
You will need to specify a path to checkpoint,
the ```transfer``` subfolder will be created in the same folder as a checkpoint.
You can find generated video there and in ```png``` subfolder loss-less verstion in '.png' format.

There are 2 principled different ways of performing transfer:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) Absolute Transfer: transfer performed using absolute postions of the driving video and appearance of the source image.
In this way there is no specific requirements for driving video and source appearance that is used.
However this usually lead to poor performance since unrelevant details such as shape is transfered.
Check transfer parameters in ```shapes.yaml``` to check how to enable this mode.

2) Realtive Transfer: from a driving video we estimate relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for transfer. This usually leads to better performance, however this require
that object in the first frame of the video and in source image being in the same pose.
Which is usually task of the user to select correct driving video and source image.
If this requirement is impossible for you task. It is better to use absolute transfer.

The approximately aligned pairs of videos is given in the data folder. (e.g  ```data/taichi.csv```).

### Image-to-video translation

In order to perform a image-to-video translation run:
```
CUDA_VISIBLE_DEVICES=0 python run.py dataset_name.yaml --mode prediction --checkpoint path/to/checkpoint
```
This will run a 3 steps:
* Estimate the keypoints from a training set
* Train rnn to predict a keypoints
* Run a predictor for each video in the dataset, starting from the first frame.
Again the ```prediction``` subfolder will be created in the same folder as a checkpoint.
You can find generated video there and in ```png``` subfolder loss-less verstion in '.png' format.

### Datasets

1) **Shapes**. This dataset is saved along with repository.
Training takes about 1 hour.

2) **Actions**. This dataset is also saved along with repository.
 And training takes about 4 hours.

3) **Nemo**. The preprocessed version of this dataset can be [downloaded]().
 Training takes about 6 hours.

4) **Taichi**. You should ask the permission to use the dataset from @sergeytulyakov.
Training takes about 9 hours.

5) **Bair**. The preprocessed version of this dataset can be [downloaded]().
Training takes about 4 hours.

6) **MGif**. The preprocessed version of this dataset can be [downloaded]().
 [Check for details on this dataset](sup-mat/MGif/README.md). Training takes about 8 hours, on 2 gpu.

7) **Vox**. The dataset can be downloaded and preprocessed using a script:
``` cd data; ./get_vox.sh ```.



#### Additional notes

Citation:

```
```
