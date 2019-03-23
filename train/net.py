# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 11:16:24
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-23 15:44:42
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np

from got10k.trackers import Tracker

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class SiameseRPN(nn.Module):

    def __init__(self, test_video=False):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(                  #1, 3, 256, 256
            #conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            #conv2
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            #conv3
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #conv4
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #conv5
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
            nn.BatchNorm2d(256)
        )

        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

        self.batcn = nn.BatchNorm2d(256)

        #self.reset_params() # we will not reset parameter

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Load Alexnet models Done' )

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)

        ckernal = self.conv1(template)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        cinput  = self.conv3(detection)


        rkernal = self.conv2(template)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        rinput  = self.conv4(detection)

        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)

        #coutput = coutput.squeeze().permute(1,2,0).reshape(-1, 2)
        #routput = routput.squeeze().permute(1,2,0).reshape(-1, 4)
        return routput, coutput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint from {}'.format(weight))

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            #nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            #nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            #nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1))
            #nn.BatchNorm2d(512))

        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

        #self.batcn = nn.BatchNorm2d(10240)
        #self.batcn2 = nn.BatchNorm2d(5120)
        #self.batcn3 = nn.BatchNorm2d(512)
        #self.batcn4 = nn.BatchNorm2d(20)
        #self.batcn5 = nn.BatchNorm2d(10)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        #kernel_reg = self.batcn(kernel_reg)

        kernel_cls = self.conv_cls_z(z)
        #kernel_cls = self.batcn2(kernel_cls)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        #x_reg = self.batcn3(x_reg)

        x_cls = self.conv_cls_x(x)
        #x_cls = self.batcn3(x_cls)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        #out_reg = self.batcn4(out_reg)
        #out_cls = self.batcn5(out_cls)
        #print('out_reg', out_reg.shape)
        #print('out_cls', out_cls.shape)

        return out_reg, out_cls


class TrackerSiamRPN(Tracker):

    def __init__(self, params, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        # setup GPU device if available
        # self.cuda   = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.params   = params

        '''setup model'''
        #self.net = SiameseRPN()
        self.net = SiamRPN()
        self.net = self.net.cuda()

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

    def step(self, ret, epoch, backward=True, update_lr=False):
        if backward:
            self.net.train()
            if update_lr:
                print(self.lr_scheduler.step())
        else:
            self.net.eval()

        cur_lr = adjust_learning_rate(self.params["lr"], self.optimizer, epoch, gamma=0.1)

        template     = ret['template_tensor'].cuda()
        detection    = ret['detection_tensor'].cuda()
        #print('template', template.shape)
        #print('detection', detection.shape)
        pos_neg_diff = ret['pos_neg_diff_tensor'].cuda()
        #print('pos_neg_diff', pos_neg_diff.shape)

        rout, cout   = self.net(template, detection)
        #print('rout', rout.shape)
        #print('cout', cout.shape)
        offsets = rout.permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().detach().numpy()
        #print('offsets', offsets.shape)
        #print('np.exp(offsets[2])', np.exp(offsets[2]), offsets[2])
        #print('self.anchors[:, 2]', self.anchors[:, 2])
        cout  = cout.squeeze().permute(1,2,0).reshape(-1, 2)
        rout  = rout.squeeze().permute(1,2,0).reshape(-1, 4)

        #print('cout, rout', cout.shape, rout.shape)

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
        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none') #1445, 4
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
