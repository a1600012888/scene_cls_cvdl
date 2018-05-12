# scene_cls_cvdl
Homework1 of computer vision and deep learning(course web: http://muyadong.com/cvdl18.html)


## Base model:

ResNeXt(50, 101, 152) and SE-ResNeXt101-32x4d which takes image of [3, 224, 224] as input

## Data preprocess and Augmentation:
code for data preprocessing and augmentation can be found in another repo of mine: my_snip

Using stadard imagenet crop for training

I used color jittery and horizontal flip


## Training

using sgd with momentum of 0.9
initial learning rate as 0.1, decrease by a factor of 10 every 3 epochs

## Data

略
