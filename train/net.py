# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import random

from got10k.trackers import Tracker

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class SiamRPNOLD(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPNOLD, self).__init__()
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

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 1, 11, 2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(1, 1, 5, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(1, 1, 3, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(1, 1, 3, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(1, 1, 3, 1),
            nn.BatchNorm2d(1))

        self.conv_reg_z = nn.Conv2d(1, 1 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(1, 1, 3)
        self.conv_cls_z = nn.Conv2d(1, 1 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(1, 1, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 1, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 1, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls

class TrackerSiamRPN(Tracker):

    def __init__(self, params, seq_dataset, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        self.video = 0
        self.frame = 0
        self.array = []

        self.seq_dataset = seq_dataset

        '''setup GPU device if available'''
        # self.cuda   = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.params   = params

        '''setup model'''
        self.net = SiamRPNOLD()
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

        index_video = range(len(self.seq_dataset))

        index_frame = len(self.seq_dataset[self.video][0])
        #print('index_video', index_video, 'index_frame', index_frame)

        if self.frame == 0:

            index_v = self.seq_dataset[self.video][0]
            self.array = []
            for i in index_v:
                self.array.append(cv2.imread(i))
            print("array", len(self.array))

            self.template_frame = self.array[self.frame]
            self.template_box   = self.seq_dataset[self.video][1][self.frame]

            self.detection_frame = self.array[self.frame+1]
            self.detection_box   = self.seq_dataset[self.video][1][self.frame+1]

            self.frame += 1

            #template_frame = Image.open(self.template_frame) #cv2.imread(self.template_frame)
            template_frame = np.asarray(self.template_frame)
            detection_frame = np.asarray(self.detection_frame)


            box = self.template_box
            box = np.array([
                box[1] - 1 + (box[3] - 1) / 2,
                box[0] - 1 + (box[2] - 1) / 2,
                box[3], box[2]], dtype=np.float32)
            self.center, self.target_sz = box[:2], box[2:]

            context = self.params['context'] * np.sum(self.target_sz)

            self.z_sz = np.sqrt(np.prod(self.target_sz + context))

            self.x_sz = self.z_sz * \
                self.params['detection_img_size'] / self.params['template_img_size']

            self.avg_color = np.mean(template_frame, axis=(0, 1))

            self.template_frame = self._crop_and_resize(template_frame, self.center, self.z_sz, self.params['template_img_size'], self.avg_color)
            #cv2.imshow('exemplar', self.template_frame)
            self.detection_frame = self._crop_and_resize(detection_frame, self.center, self.x_sz, self.params['detection_img_size'], self.avg_color)

            self.template_frame = torch.from_numpy(self.template_frame).permute([2, 0, 1]).unsqueeze(0).float()

        else:
            self.detection_frame = self.array[self.frame+1]
            self.detection_box   = self.seq_dataset[self.video][1][self.frame+1]

            self.frame += 1
            #detection_frame = Image.open(self.detection_frame) #cv2.imread(self.detection_frame)
            self.detection_frame = np.asarray(self.detection_frame)

            #print('detection_frame', detection_frame.shape)

            box = self.detection_box
            box = np.array([
                box[1] - 1 + (box[3] - 1) / 2,
                box[0] - 1 + (box[2] - 1) / 2,
                box[3], box[2]], dtype=np.float32)
            self.center, self.target_sz = box[:2], box[2:]

            context = self.params['context'] * np.sum(self.target_sz)

            self.z_sz = np.sqrt(np.prod(self.target_sz + context))

            self.x_sz = self.z_sz * \
                self.params['detection_img_size'] / self.params['template_img_size']

            self.avg_color = np.mean(self.detection_frame, axis=(0, 1))

            self.detection_frame = self._crop_and_resize(self.detection_frame, self.center, self.x_sz, self.params['detection_img_size'], self.avg_color)
            if self.frame+1 == index_frame:
                self.video = random.choice(index_video)
                self.frame = 0

        ret = data_loader.__get__( self.template_box)#self.detection_box)

        cur_lr = adjust_learning_rate(self.params["lr"], self.optimizer, epoch, gamma=0.1)

        #template     = ret['template_tensor']#.cuda()
        #detection    = ret['detection_tensor']#.cuda()
        pos_neg_diff = ret['pos_neg_diff_tensor']#.cuda()

        self.detection_frame = torch.from_numpy(self.detection_frame).permute([2, 0, 1]).unsqueeze(0).float()

        rout, cout   = self.net(self.template_frame, self.detection_frame)

        offsets = rout.permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().detach().numpy()

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

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        #print('npad', npad)
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        #cv2.imshow('image',image)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]
        #cv2.imshow('image2',patch)

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))
        #cv2.imshow('image3',patch)

        return patch

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):

        cout, rout = predictions

        """ class """
        class_pred, class_target = cout, targets[:, 0].long()
        #print('class_target', class_target)

        pos_index , neg_index    = list(np.where(class_target.cpu() == 1)[0]), list(np.where(class_target.cpu() == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, reduction='none')
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred = rout
        #print('reg_pred', reg_pred)
        reg_target = targets[:, 1:]
        #print('reg_target', reg_target)

        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none') #1445, 4
        #print('rloss1', rloss)
        rloss = torch.div(torch.sum(rloss, dim = 1), 4)
        #print('rloss2', rloss)
        #print('pos_index', pos_index)

        #print('torch.sum(rloss[pos_index])', torch.sum(rloss[pos_index]))

        rloss = torch.div(torch.sum(rloss[pos_index]), 16)
        #print('rloss3', rloss)


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
