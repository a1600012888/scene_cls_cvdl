#!/usr/bin/env mdl
from SEResNeXt import SEResNeXt, init_trained_weights, init_trained_weights_parall
import torch
import numpy as np
from torch.autograd import Variable
from DataSet import Dataset
import my_snip.metrics as metrics
from my_snip.clock import  TrainClock, AvgMeter
from my_snip.config import MultiStageLearningRatePolicy
import torch.nn as nn
import torch.optim as optim

from TorchDataset import TorchDataset
from torch.utils.data import DataLoader
import time

from tensorboardX import SummaryWriter


# In[25]:
num_workers = 8
minibatch_size = 16
num_epoch = 5

ValAccuracy = []


model_path = './training_models/p_se_resnext101_32x4d_exp6.pt'

def append_answr(info_path, model_path, answer):
    with open(info_path, 'a') as f:
        st = [model_path, ' --- ', str(answer), '\n']
        f.writelines(st)

def single_crop_eval(model_path, multi_gpu = True, gpu_id = 0, info_path = './training_models/answer.txt'):
    ds_val = TorchDataset('validation')

    dl_val = DataLoader(ds_val, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)

    net = SEResNeXt(101, num_class=80)
    if multi_gpu:
        init_trained_weights_parall(net, model_path)
    else:
        init_trained_weights(net, model_path)
    net.cuda(gpu_id)

    # In[22]:
    ud_loss_m = AvgMeter('ud_loss')
    accuracy_m = AvgMeter('top-1-accuracy')
    top3_accuracy_m = AvgMeter('top-3-accuracy')

    criterion = nn.CrossEntropyLoss().cuda(gpu_id)

    net.eval()
    print('Begin validation!')

    ud_loss_m.reset()
    accuracy_m.reset()
    top3_accuracy_m.reset()
    for i, mn_batch in enumerate(dl_val):

      #  if i > 401:
       #     break
        data = mn_batch['data'].type(torch.FloatTensor)
        label = mn_batch['label'].type(torch.LongTensor).squeeze_()
        inp_var = Variable(data).cuda(gpu_id)
        label_var = Variable(label).cuda(gpu_id)


        pred = net.forward(inp_var)

        ud_loss = criterion(pred, label_var)

        acc, t3_acc = metrics.torch_accuracy(pred.data, label_var.data, (1, 3))

        ud_loss_m.update(ud_loss.data[0], inp_var.size(0))
        accuracy_m.update(acc[0], inp_var.size(0))
        top3_accuracy_m.update(t3_acc[0], inp_var.size(0))

    print('Validation Done!')
    print('ud_loss: {}, accuracy: {}, top-3-accuracy {}'.format(ud_loss_m.mean, accuracy_m.mean, top3_accuracy_m.mean))

    append_answr(info_path, model_path, [ud_loss_m.mean, accuracy_m.mean, top3_accuracy_m.mean])

# vim: ts=4 sw=4 sts=4 expandtab
