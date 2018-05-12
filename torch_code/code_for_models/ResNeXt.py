
# coding: utf-8

# In[66]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


# In[92]:


class ResNextBoottlenexk(nn.Module):
    
    ##input_chn, cardinality, base_width, group_width
    
    def __init__(self, input_chn, cardinality, width, output_chn, stride = 1, 
                 has_proj = False, prefix = 'Bottleneck'):
        super(ResNextBoottlenexk, self).__init__()
        
        
        
        C = cardinality
        D = int(width / cardinality)
        
        self.proj = None
        if has_proj:
            
            self.proj = nn.Sequential(
                nn.Conv2d(input_chn, output_chn, kernel_size=1, stride=stride, padding = 0, bias = False), 
                nn.BatchNorm2d(output_chn), )
            
        print(prefix + ':', 'input_chn: {}, output_chn: {}, cardinality: {}'.format(input_chn, output_chn, cardinality))
        
        self.conv_bottle = nn.Conv2d(input_chn, width, kernel_size=1, stride=1, padding=0, bias = False)
        print('conv_bottle {}'.format(list(self.conv_bottle.weight.size())))
        self.bn_bottle = nn.BatchNorm2d(width)
        #print('bn_bottle  and  relu_bottle')
        
        self.conv_group = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding = 1, bias=False, groups=cardinality)
        print('conv_group {}'.format(list(self.conv_group.weight.size())))
        self.bn_group = nn.BatchNorm2d(D*C)
        #print('bn_group  and  relu_group')
        
        self.conv_expand = nn.Conv2d(D*C, output_chn, kernel_size=1, stride=1, padding=0, bias = False)
        print('conv_expand {}'.format(list(self.conv_expand.weight.size())))
        self.bn_expand = nn.BatchNorm2d(output_chn)
        #print('bn_expand')
        
    def forward(self, x):
        #inp_shape = list(x.size())
        
        #assert inp_shape[1] == self.input_chn, inp_shape
        
        #print(inp_shape)
        
        proj = x
        
        y = self.conv_bottle(x)
        y = self.bn_bottle(y)
        y = F.relu(y)
        
        
        y = self.conv_group(y)
        y = F.relu(self.bn_bottle(y))
        
        y = self.conv_expand(y)
        y = self.bn_expand(y)
        
        if self.proj is not None:
            proj = self.proj(proj)
        
        y = proj + y
        y = F.relu(y)
        
        return y
    
        


# In[93]:


def make_stage(input_chn, width, block_num, cardinality, output_chn, proj_stride = 2, name = 'stage_2'):
    
    blocks = []
    
    blocks.append(ResNextBoottlenexk(input_chn, cardinality, width, output_chn, 
                                     proj_stride, True, prefix = '{}_{}'.format(name, '1'))
                 )
    
    for i in range(1, block_num):
        
        blocks.append(ResNextBoottlenexk(output_chn, cardinality, width, 
                                         output_chn, prefix = "{}_{}".format(name, str(i + 1))))
        
    return nn.Sequential(*blocks)


# In[106]:


class ResNeXt(nn.Module):
    
    stage_blcoks = [3, 4, 6, 3]
    
    def __init__(self, depth, cardinality = 32, base_width = 128, num_class = 1000):
        
        super(ResNeXt, self).__init__()
        
        
        assert depth in [50, 101, 152], 'Depth should be in [50, 101, 152]!'
        
        assert base_width % cardinality == 0, 'base_width should be a multiple of cardinality'
        
        if depth == 152:
            self.stage_blcoks[1] = 8
        self.stage_blcoks[2] = (depth - (np.sum(self.stage_blcoks) - 6)*3 - 2) // 3
        
        
        self.conv_1_7x7 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage_2 = make_stage(64, base_width, self.stage_blcoks[0], cardinality, 
                                  output_chn=256, proj_stride = 1, name = 'stage_2')
        
        self.stage_3 = make_stage(256, base_width * 2, self.stage_blcoks[1], cardinality, 
                                  512, 2, name = 'stage_3')
        
        self.stage_4 = make_stage(512, base_width * 4, self.stage_blcoks[2], cardinality, 
                                  1024, 2, name = 'stage_4')
        
        self.stage_5 = make_stage(1024, base_width * 8, self.stage_blcoks[3], cardinality, 
                                  2048, 2, name = 'stage_5')
        
        
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7)
        
        self.fc = nn.Linear(2048, num_class)
        
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
                
                
    def forward(self, inp_var):
        
        y = self.conv_1_7x7(inp_var)
        y = self.bn_1(y)
        y = F.relu(y)
        
        y = self.max_pool(y)
        
        y = self.stage_2(y)
        y = self.stage_3(y)
        y = self.stage_4(y)
        y = self.stage_5(y)
        
        y = self.global_avg_pool(y)
        #print(list(y.size()))
        
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        
        #print(list(y.size()))
        return y
        
        
        
        


# In[110]:


def test():
    
    net = ResNeXt(50)
    test_img = torch.randn(*[1, 3, 224, 224])
    test_img = Variable(test_img)
    pred = net.forward(test_img)

