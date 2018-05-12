#!/usr/bin/env mdl
from single_crop_val import single_crop_eval
import os
'''
model_paths = [
    './training_models/p_se_resnext101_32x4d_exp4.pt',
    './training_models/p_se_resnext101_32x4d_exp5.pt',
    './training_models/p_se_resnext101_32x4d_exp6.pt',
    './training_models/se_resnext101_32x4d_exp3.pt',
    './training_models/se_resnext101_32x4d_exp4.pt',
    './training_models/se_resnext101_32x4d_exp5.pt',
]
'''
model_paths = [
    './training_models/se_resnext101_32x4d_fine_tune.pt',
    './training_models/se_resnext101_32x4d_fine_tune1.pt',
    './training_models/se_resnext101_32x4d_fine_tune2.pt',
]
multi_gpu = [
    False,
    False,
    False
]
for path, gpu in zip(model_paths, multi_gpu):
    print('evaluating {}!'.format(path))
    single_crop_eval(path, gpu)
# vim: ts=4 sw=4 sts=4 expandtab
