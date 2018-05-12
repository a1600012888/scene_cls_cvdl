
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from collections import OrderedDict


# In[2]:


class SEModule(nn.Module):

    def __init__(self, chn, reduc):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(chn, chn // reduc, kernel_size=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(chn // reduc, chn, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        inp = x
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)

        y = self.fc2(y)
        y = self.sigmoid(y)

        y = inp * y
        return y


# In[55]:


class SEResNeXtBottleneck(nn.Module):


    def __init__(self, input_chn, cardinality, width, output_chn, reduction = 16,
                 stride = 1, has_proj = False, prefix = 'Bottleneck'):
        super(SEResNeXtBottleneck, self).__init__()



        C = cardinality
        D = int(width / cardinality)
        print(prefix + ':', 'input_chn: {}, output_chn: {}, cardinality: {}'.format(input_chn, output_chn, cardinality))


        self.conv1 = nn.Conv2d(input_chn, width, kernel_size=1, stride=1, padding=0, bias = False)

        self.bn1 = nn.BatchNorm2d(width)


        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding = 1, bias=False, groups=cardinality)

        self.bn2 = nn.BatchNorm2d(D*C)


        self.conv3 = nn.Conv2d(D*C, output_chn, kernel_size=1, stride=1, padding=0, bias = False)

        self.bn3 = nn.BatchNorm2d(output_chn)

        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(output_chn, reduction)

        self.downsample = None
        if has_proj:

            self.downsample = nn.Sequential(
                nn.Conv2d(input_chn, output_chn, kernel_size=1, stride=stride, padding = 0, bias = False),
                nn.BatchNorm2d(output_chn), )

    def forward(self, x):
        proj = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        y = self.se_module(y)

        if self.downsample is not None:
            proj = self.downsample(proj)

        y = y + proj
        y = self.relu(y)

        return y


# In[133]:


class SEResNeXt(nn.Module):
    stage_blcoks = [3, 4, 6, 3]

    def __init__(self, depth, reduction = 16, cardinality = 32, base_width = 128, num_class = 1000):
        super(SEResNeXt, self).__init__()

        assert depth in [50, 101, 152], 'Depth should be in [50, 101, 152]!'
        assert base_width % cardinality == 0, 'base_width should be a multiple of cardinality'

        if depth == 152:
            self.stage_blcoks[1] = 8
            self.stage_blcoks[2] = 36
        if depth == 101:
            self.stage_blcoks[2] = 23
        #self.stage_blcoks[2] = (depth - (np.sum(self.stage_blcoks) - 6)*3 - 2) // 3


        layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(3, 2, ceil_mode=True))
            ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self.make_layer(64, base_width, self.stage_blcoks[0], cardinality,
                                  output_chn=256, reduction=reduction, proj_stride = 1, name = 'stage_2')

        self.layer2 = self.make_layer(256, base_width * 2, self.stage_blcoks[1], cardinality,
                                  512, reduction, 2, name = 'stage_3')

        self.layer3 = self.make_layer(512, base_width * 4, self.stage_blcoks[2], cardinality,
                                  1024, reduction, 2, name = 'stage_4')

        self.layer4 = self.make_layer(1024, base_width * 8, self.stage_blcoks[3], cardinality,
                                  2048, reduction, 2, name = 'stage_5')

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.last_linear = nn.Linear(2048, num_class)

        self.my_kaiming_init()

    def my_kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print('conv: ')
                #print(list(m.weight.size()))
                n = m.kernel_size[0]*m.kernel_size[1]*m.in_channels

                m.weight.data.normal_(0, math.sqrt(2.0/n))

            if isinstance(m, nn.Linear):
                #print('fc: ')
                #print(list(m.weight.size()))
                nn.init.kaiming_normal(m.weight)

                m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                #print('bn!')
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, input_chn, width, block_num, cardinality, output_chn,
                   reduction = 16, proj_stride = 2, name = 'stage_2_'):
        blocks = []

        blocks.append(SEResNeXtBottleneck(input_chn, cardinality, width, output_chn, reduction,
                                         proj_stride, True, prefix = '{}_{}'.format(name, '1'))
                     )

        for i in range(1, block_num):

            blocks.append(SEResNeXtBottleneck(output_chn, cardinality, width,
                                             output_chn, reduction, prefix = "{}_{}".format(name, str(i + 1))))

        return nn.Sequential(*blocks)

    def features(self, x):
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


# In[134]:


def init_pretrained_weights_1000(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    own_state = model.state_dict()

    #for name, var in state_dict.items():
     #   print(name)
    #
    if 'layer3.6.conv1.weight' in own_state:
        print('WHY??')
    for name, param in state_dict.items():
        if name not in ['last_linear.weight', 'last_linear.bias']:

            print(name)
            if name in own_state:
                print('!!')

            own_state[name].copy_(param)

def init_trained_weights(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))

def init_trained_weights_parall(model, state_dict_path):
    state_dict = torch.load(state_dict_path)

    own_state = model.state_dict()
    for name, var in state_dict.items():
        rname = name[7:]
        #print(name, rname)
        own_state[rname].copy_(var)


# In[137]:


def test():

    net = SEResNeXt(101, num_class=80)

    dic = net.state_dict()
    if 'layer3.6.conv1.weight' in dic:
        print('WHY??')
    #init_pretrained_weights_1000(net, '../trained_models/se_resnext101_32x4d.pt')
    init_trained_weights(net, './se_resnext101_32x4d.pt')

    test_img = torch.randn(*[1, 3, 224, 224])
    test_img = Variable(test_img)
    pred = net.forward(test_img)

    #torch.save(dic, './se_resnext101_32x4d.pt')


# In[138]:


if __name__ == '__main__':
    test()


# In[120]:


'''
net = SEResNeXt(101, num_class=80)
dic = net.state_dict()
for name, var in dic.items():
    print(name)
'''

