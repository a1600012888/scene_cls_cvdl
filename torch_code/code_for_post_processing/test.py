
# coding: utf-8

# In[2]:


from SEResNeXt import SEResNeXt, init_trained_weights, init_pretrained_weights_1000, init_trained_weights_parall
import torch
import numpy as np
from torch.autograd import Variable
from DataSet import Dataset

import my_snip.metrics as metrics
from my_snip.clock import  TrainClock, AvgMeter
import torch.nn as nn

from TorchDataset import TorchDataset
from torch.utils.data import DataLoader
import time
from TestDataset import TestTorchDataset


def write_result(info_path, result, target_path):
    f1 = open(info_path)
    f2 = open(target_path, 'w+')
    i = 0
    for line in f1:
        name = line[5:-1]
        top3 = result[name]
        f2.write(line[:-1]+' '+ str(top3[0])+' '
                +str(top3[1])+' '
                +str(top3[2]))
        f2.write('\n')
        i+=1
        print('writing {}'.format(name))
    print(i, 'test result')


def vote(output):
    _, pred = output.topk(3, 1, True, True)
    a = np.zeros(shape = (80), dtype=np.int8)
    preds = pred.numpy()

    for i in preds:

        a[i] += 1
    #print(a)
    a = np.argsort(a)

    #print('Final! {}'.format(a[-3:]))
    return a[-3:]


def write_test(model_path, target_path, multi_gpu = True, gpu_id = 0, test_info_path = '../data/test.info'):

    crop_time = 10
    minibatch_size = 32


    ds_test = TestTorchDataset(crop_time=10)


    net = SEResNeXt(101, num_class=80)
    #init_trained_weights(net, model_path)
    #init_pretrained_weights_1000(net, model_path)
    if multi_gpu:
        init_trained_weights_parall(net, model_path)
    else:
        init_trained_weights(net, model_path)
    net.cuda(gpu_id)


    ans_dict = {}
    net.eval()

    for i in range(ds_test.instance_per_epoch):

        sample = ds_test[i]

        imgs = sample['data'].type(torch.FloatTensor)
        name = sample['name']

        inp_var = Variable(imgs).cuda(gpu_id)

        output = net.forward(inp_var)

        pred = vote(output.cpu().data)

        ans_dict[name] = pred

    print('Done infenrence! Begin writing!')

    write_result(test_info_path, ans_dict, target_path)


# In[11]:


