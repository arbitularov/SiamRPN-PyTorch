from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker
from data_loader import TrainDataLoader

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

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        #self.net = SiameseRPN()
        self.net = SiamRPN()
        self.data_loader = TrainDataLoader(self.net, params)

        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

    def init(self, image, box):
        #image = np.asarray(image)

        self.box = box

        ret = self.data_loader.get_template(image, box)

        self.anchors = ret['anchors']

        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(ret['template_tensor'])

    def update(self, detection):

        ret = self.data_loader.__get__(detection, self.box)

        detection_tensor    = ret['detection_tensor']#.cuda()

        with torch.set_grad_enabled(False):
            self.net.eval()
            rout, cout = self.net.inference(detection_tensor, self.kernel_reg, self.kernel_cls)

        cout = cout.reshape(-1, 2)
        rout = rout.reshape(-1, 4)
        cout = cout.cpu().detach().numpy()
        score = 1/(1 + np.exp(cout[:,1]-cout[:,0]))
        diff   = rout.cpu().detach().numpy() #1445

        num_proposals = 1
        score_64_index = np.argsort(score)[::-1][:num_proposals]
        #print('score_64_index', score_64_index)

        score64 = score[score_64_index]
        diffs64 = diff[score_64_index, :]
        anchors64 = self.anchors[score_64_index]
        proposals_x = (anchors64[:, 0] + anchors64[:, 2] * diffs64[:, 0]).reshape(-1, 1)
        proposals_y = (anchors64[:, 1] + anchors64[:, 3] * diffs64[:, 1]).reshape(-1, 1)
        proposals_w = (anchors64[:, 2] * np.exp(diffs64[:, 2])).reshape(-1, 1)
        proposals_h = (anchors64[:, 3] * np.exp(diffs64[:, 3])).reshape(-1, 1)
        box = np.hstack((proposals_x, proposals_y, proposals_w, proposals_h))
        self.box = box[0]
        print("box", box[0])

        return box[0]
