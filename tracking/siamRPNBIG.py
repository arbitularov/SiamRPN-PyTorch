import cv2
import torch
import numpy as np
import torch.nn as nn
from util import Util as util
import torch.nn.functional as F
from config import TrackerConfig
from torch.autograd import Variable
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

class TrackerSiamRPNBIG(Tracker):
    def __init__(self, params, net_path = None, **kargs):
        super(TrackerSiamRPNBIG, self).__init__(name='SiamRPN', is_deterministic=True)

        '''setup model'''
        self.net = SiamRPN()
        self.data_loader = TrainDataLoader(self.net, params)

        '''setup GPU device if available'''
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        if self.cuda:
            self.net.cuda()

        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))

        self.net.eval()

    def init(self, target_img, target_box):
        self.box = target_box
        target_imgX = target_img

        target_img = np.asarray(target_img)

        target_centor, target_size = util.x1y1_wh_to_xy_wh(target_box) # x1y1wh -> xywh # convert to bauding box centor

        self.state = dict()
        p = TrackerConfig()
        self.state['target_img_h'] = target_img.shape[0]
        self.state['target_img_w'] = target_img.shape[1]

        if ((target_size[0] * target_size[1]) / float(self.state['target_img_h'] * self.state['target_img_w'])) < 0.004:
            p.detection_size = 287  # small object big search region

        p.score_size = int((p.detection_size - p.target_size) / p.total_stride + 1)

        p.anchor = util.generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)


        avg_chans = np.mean(target_img, axis=(0, 1))

        wc_z = target_size[0] + p.context_amount * sum(target_size)
        hc_z = target_size[1] + p.context_amount * sum(target_size)
        s_z = round(np.sqrt(wc_z * hc_z))

        # initialize the exemplar
        z_crop = util.get_subwindow_tracking(target_img, target_centor, p.target_size, s_z, avg_chans)

        ret = self.data_loader.get_template(target_imgX, self.box)

        z = Variable(z_crop.unsqueeze(0))
        self.kernel_reg, self.kernel_cls = self.net.learn(ret['template_tensor'])#.cuda())
        #self.kernel_reg, self.kernel_cls = self.net.learn(z)#.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)

        self.state['p'] = p
        self.state['avg_chans'] = avg_chans
        self.state['window'] = window
        self.state['target_centor'] = target_centor
        self.state['target_size'] = target_size

    def update(self, im, iter=0):
        ret = self.data_loader.__get__(im, self.box)
        self.detection_tensor = ret['detection_tensor']

        im = np.asarray(im)
        p = self.state['p']
        avg_chans = self.state['avg_chans']
        window = self.state['window']
        target_pos = self.state['target_centor']
        target_sz = self.state['target_size']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.target_size / s_z # target_size
        d_search = (p.detection_size - p.target_size) / 2 # detection_size
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(util.get_subwindow_tracking(im, target_pos, p.detection_size, round(s_x), avg_chans).unsqueeze(0))

        #target_pos, target_sz, score = self.tracker_eval(self.net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
        target_pos, target_sz, score = self.tracker_eval(self.net, x_crop, target_pos, target_sz * scale_z, window,
                                                    scale_z, p)
        target_pos[0] = max(0, min(self.state['target_img_w'], target_pos[0]))
        target_pos[1] = max(0, min(self.state['target_img_h'], target_pos[1]))
        target_sz[0] = max(10, min(self.state['target_img_w'], target_sz[0]))
        target_sz[1] = max(10, min(self.state['target_img_h'], target_sz[1]))
        #self.state['target_centor'] = target_pos
        #self.state['target_size'] = target_sz
        #self.state['score'] = score

        #res = cxy_wh_2_rect(self.state['target_centor'], self.state['target_size'])
        res = util.cxy_wh_2_rect(target_pos, target_sz)

        self.box = res

        return res

    def tracker_eval(self, net, x_crop, target_pos, target_sz, window, scale_z, p):
        delta, score = net.inference(self.detection_tensor, self.kernel_reg, self.kernel_cls)
        #delta, score = net.inference(x_crop, self.kernel_reg, self.kernel_cls)

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

        delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        best_pscore_id = np.argmax(pscore)

        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        return target_pos, target_sz, score[best_pscore_id]
