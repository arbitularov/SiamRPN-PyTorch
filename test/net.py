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
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
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

        coutput = coutput.squeeze().permute(1,2,0).reshape(-1, 2)
        routput = routput.squeeze().permute(1,2,0).reshape(-1, 4)
        return coutput, routput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint from {}'.format(weight))


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        # setup GPU device if available
        #self.cuda = torch.cuda.is_available()
        #self.device = torch.device('cuda:0' if self.cuda else 'cpu')


        '''setup model'''
        self.net = SiameseRPN()

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        #self.net = self.net.to(self.device)

        '''setup optimizer'''

        self.criterion = MultiBoxLoss()

        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay = 5e-5)

    def init(self, image, box):
        image = np.asarray(image)
        #print('image_init', image.shape)
        #print('box', box)
        self.box = box

    def update(self, image):
        image = np.asarray(image)

        #print('image_update', image.shape)
        box = self.box

        return box

    def step(self, ret, epoch, backward=True, update_lr=False):
        if backward:
            self.net.train()
            if update_lr:
                print(self.lr_scheduler.step())
        else:
            self.net.eval()

        cur_lr = adjust_learning_rate(0.001, self.optimizer, epoch, gamma=0.1)

        template = ret['template_tensor']#.cuda()
        detection= ret['detection_tensor']#.cuda()
        pos_neg_diff = ret['pos_neg_diff_tensor']#.cuda()
        cout, rout = self.net(template, detection)
        predictions, targets = (cout, rout), pos_neg_diff
        closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index = self.criterion(predictions, targets)

        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item() #closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index, cur_lr


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
    detection= torch.ones((1, 3, 256, 256))

    y1, y2 = model(template, detection)
    print(y1.shape) #[1, 10, 17, 17]
    print(y2.shape) #[1, 20, 17, 17]15
