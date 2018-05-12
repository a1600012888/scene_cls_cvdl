#!/usr/bin/env mdl
from test import write_test

model_path = [
    './training_models/p_se_resnext101_32x4d_exp4.pt',
    './training_models/p_se_resnext101_32x4d_exp5.pt',
    './training_models/p_se_resnext101_32x4d_exp6.pt',
    './training_models/se_resnext101_32x4d_exp3.pt',
    './training_models/se_resnext101_32x4d_exp4.pt',
    './training_models/se_resnext101_32x4d_exp5.pt',
    './training_models/se_resnext101_32x4d_fine_tune.pt',
    './training_models/se_resnext101_32x4d_fine_tune1.pt',
    './training_models/se_resnext101_32x4d_fine_tune2.pt',
]
gpu = [
    True,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
]
target_path = [
    '../data/answer1.info',
    '../data/answer2.info',
    '../data/answer3.info',
    '../data/answer4.info',
    '../data/answer5.info',
    '../data/answer6.info',
    '../data/answer7.info',
    '../data/answer8.info',
    '../data/answer9.info',
]

gpu_id = 3

for m, g, t in zip(model_path[8:], gpu[8:], target_path[8:]):
    print('Testing {}'.format(m))
    print('Writing {}'.format(t))
    write_test(m, t, g, gpu_id)

# vim: ts=4 sw=4 sts=4 expandtab
