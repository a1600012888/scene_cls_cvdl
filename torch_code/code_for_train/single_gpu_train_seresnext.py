
# coding: utf-8

# In[24]:


from SEResNeXt import SEResNeXt, init_trained_weights
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
gpu_id = 3
num_workers = 8
minibatch_size = 16
num_epoch = 16

ValAccuracy = []

learning_rate_policy = [[3, 0.01],
                        [4, 0.001],
                        [4, 0.0001],
                        [7, 0.00002]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)


model_path = './training_models/se_resnext101_32x4d.pt'
save_path = './net6.pt'
best_path = './training_models/se_resnext101_32x4d_exp6.pt'
log_dir = './logs/se_resnext101_32x4d_exp6'
json_path = log_dir + '/' + 'all_scalars.json'
writer = SummaryWriter(log_dir)



# In[19]:


def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr



# In[20]:


ds_train = TorchDataset('train')
ds_val = TorchDataset('validation')

dl_train = DataLoader(ds_train, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
dl_val = DataLoader(ds_val, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)


step_per_epoch = ds_train.instance_per_epoch // minibatch_size


# In[14]:


net = SEResNeXt(101, num_class=80)
init_trained_weights(net, model_path)

net.cuda(gpu_id)


# In[22]:


criterion = nn.CrossEntropyLoss().cuda(gpu_id)

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay=1e-4)
#optimizer = optim.Adam(net.parameters(), lr = 0.01, weight_decay = 1e-4)
clock = TrainClock()

ud_loss_m = AvgMeter('ud_loss')
accuracy_m = AvgMeter('top-1-accuracy')
top3_accuracy_m = AvgMeter('top-3-accuracy')
data_time_m = AvgMeter('Reading Batch Data')
batch_time_m = AvgMeter('Batch time')

for epoch_i in range(num_epoch):

    net.train()
    print('Epoch {} starts'.format(epoch_i))
    clock.tock()
    epoch_time = time.time()

    adjust_learning_rate(optimizer, clock.epoch)

    ud_loss_m.reset()
    accuracy_m.reset()
    top3_accuracy_m.reset()
    data_time_m.reset()
    batch_time_m.reset()

    start_time = time.time()

    for i, mn_batch in enumerate(dl_train):

        #if i % 50 == 0:
         #   print(i)
        #print(i)
        #if i > 201:
         #   break
        data_time_m.update(time.time() - start_time)

        clock.tick()
        data = mn_batch['data'].type(torch.FloatTensor)
        label = mn_batch['label'].type(torch.LongTensor).squeeze_()

        inp_var = Variable(data).cuda(gpu_id)
        label_var = Variable(label).cuda(gpu_id)

        optimizer.zero_grad()

        pred = net.forward(inp_var)

        #print(label_var.size())
        ud_loss = criterion(pred, label_var)

        acc, t3_acc = metrics.torch_accuracy(pred.data, label_var.data, (1, 3))

        writer.add_scalar('Train/un_decay_loss', ud_loss.data[0], clock.step)
        writer.add_scalar('Trian/top_acc', acc[0], clock.step)
        writer.add_scalar('Trian/top_3_acc', t3_acc[0], clock.step)

        ud_loss_m.update(ud_loss.data[0], inp_var.size(0))
        accuracy_m.update(acc[0], inp_var.size(0))

        top3_accuracy_m.update(t3_acc[0], inp_var.size(0))

        ud_loss.backward()

        optimizer.step()

        batch_time_m.update(time.time() - start_time)
        start_time = time.time()

        if clock.minibatch % 200 == 0:
            print("step {} :".format(clock.minibatch), ud_loss.data[0], acc[0], t3_acc[0])
            print('data time: {} mins, batch time: {} mins'.format(data_time_m.mean, batch_time_m.mean))
            print('Epoch{} time: {} mins.  {} mins to run'.format(clock.epoch, (start_time - epoch_time)/60,
                                                                (batch_time_m.mean * (step_per_epoch - clock.minibatch) / 60)))

    print('Epoch {} Finished! Lasting {} mins'.format(clock.epoch, (start_time - epoch_time)/60))
    print('ud_loss: {}, accuracy: {}, top-3-accuracy {}'.format(ud_loss_m.mean, accuracy_m.mean, top3_accuracy_m.mean))

    writer.add_scalar('Time/Epoch-time', (start_time - epoch_time)/60)
    writer.add_scalar('Train/Epoch-un_decay_loss', ud_loss_m.mean, clock.epoch)
    writer.add_scalar('Trian/Epoch-top_acc', accuracy_m.mean, clock.epoch)
    writer.add_scalar('Trian/Epoch-top_3_acc', top3_accuracy_m.mean, clock.epoch)

    torch.save(net.state_dict(), save_path)

    if clock.epoch % 1 == 0:

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

        writer.add_scalar('Val/Epoch-un_decay_loss', ud_loss_m.mean, clock.epoch)
        writer.add_scalar('Val/Epoch-top_acc', accuracy_m.mean, clock.epoch)
        writer.add_scalar('Val/Epoch-top_3_acc', top3_accuracy_m.mean, clock.epoch)

        ValAccuracy.append(accuracy_m.mean)
        if accuracy_m.mean >= max(ValAccuracy):
            torch.save(net.state_dict(), best_path)


        print('Best top-1-accuracy: {}, saved in {}'.format(max(ValAccuracy), best_path))

#    torch.save(net.state_dict(), './net.pt')
writer.export_scalars_to_json(json_path)
writer.close()

print('Training Finished')
torch.save(net.state_dict(), save_path)

print('model saved in {}'.format(save_path))
print('Best top-1-accuracy: {}, saved in {}'.format(max(ValAccuracy), best_path))
