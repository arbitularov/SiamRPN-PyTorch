# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from loss import MultiBoxLoss
import matplotlib.pyplot as plt
import torch.nn.functional as F
from parameters import Config as config
from got10k.trackers import Tracker

class SiamRPN(nn.Module):

    def __init__(self, anchor_num = 5):
        super(SiamRPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))

        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * self.anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3) # 8 is batch_size
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3) # 8 is batch_size
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num*1, 1)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k) # 8 is batch_size
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k) # 8 is batch_size

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls

class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        '''setup GPU device if available'''
        # self.cuda   = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        '''setup model'''
        self.net = SiamRPN()
        #self.net = self.net.cuda()

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location = lambda storage, loc: storage ))
        #self.net = self.net.to(self.device)

        '''setup optimizer'''
        self.criterion   = MultiBoxLoss()

        self.optimizer   = torch.optim.SGD(
            self.net.parameters(),
            lr           = config.lr,
            momentum     = config.momentum,
            weight_decay = config.weight_decay)

    def step(self, epoch, data_loader, example, index_list, backward=True):

        if backward:
            self.net.train()
        else:
            self.net.eval()

        cur_lr = adjust_learning_rate(config.lr, self.optimizer, epoch, gamma=0.1)

        template, detection, pos_neg_diff = data_loader.__getitem__(random.choice(index_list))

        rout, cout = self.net(template, detection)

        cout = cout.squeeze().permute(1,2,0).reshape(-1, 2) # 8 is batch_size

        rout = rout.squeeze().permute(1,2,0).reshape(-1, 4) # 8 is batch_size

        predictions, targets = (cout, rout), pos_neg_diff

        closs, rloss, loss = self.criterion(predictions, targets)

        if backward:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        return closs, rloss, loss, cur_lr

    '''save model'''
    def save(self,model, exp_name_dir, epoch):
        model_save_dir_pth = '{}/model'.format(exp_name_dir)
        if not os.path.exists(model_save_dir_pth):
                os.makedirs(model_save_dir_pth)
        net_path = os.path.join(model_save_dir_pth, 'model_e%d.pth' % (epoch + 1))
        torch.save(model.net.state_dict(), net_path)

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
