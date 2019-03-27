# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from got10k.trackers import Tracker

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
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

        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls

class TrackerSiamRPN(Tracker):

    def __init__(self, params, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        '''setup GPU device if available'''
        # self.cuda   = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.params   = params

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
            lr           = self.params["lr"],
            momentum     = self.params["momentum"],
            weight_decay = self.params["weight_decay"])

    def step(self, data_loader, epoch, backward=True):

        if backward:
            self.net.train()
        else:
            self.net.eval()

        ret = data_loader.__get__()

        cur_lr = adjust_learning_rate(self.params["lr"], self.optimizer, epoch, gamma=0.1)

        template     = ret['template_frame_tensor']#.cuda()
        detection    = ret['detection_frame_tensor']#.cuda()
        pos_neg_diff = ret['pos_neg_diff_tensor']#.cuda()

        rout, cout   = self.net(template, detection)

        offsets = rout.permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().detach().numpy()

        cout  = cout.squeeze().permute(1,2,0).reshape(-1, 2)
        rout  = rout.squeeze().permute(1,2,0).reshape(-1, 4)

        predictions, targets = (cout, rout), pos_neg_diff
        closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index = self.criterion(predictions, targets)

        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index, cur_lr #loss.item()

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):

        cout, rout = predictions

        """ class """
        class_pred, class_target = cout, targets[:, 0].long()

        pos_index , neg_index    = list(np.where(class_target.cpu() == 1)[0]), list(np.where(class_target.cpu() == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, reduction='none')
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred = rout
        reg_target = targets[:, 1:]

        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none')

        rloss = torch.div(torch.sum(rloss, dim = 1), 4)

        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        loss = closs + rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    model = SiameseRPN()

    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 271, 271))

    y1, y2 = model(template, detection)
    print(y1.shape) #[1, 10, 19, 19]
    print(y2.shape) #[1, 20, 19, 19]15
