import cv2
import torch
import numpy as np
import torch.nn as nn
from util import Util as util
import torch.nn.functional as F
from config import TrackerConfig
from config import config
from torch.autograd import Variable
from got10k.trackers import Tracker
from network import SiameseAlexNet
from data_loader import TrainDataLoader
from PIL import Image, ImageOps, ImageStat, ImageDraw

class SiamRPN(nn.Module):

    def __init__(self, anchor_num = 5):
        super(SiamRPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size = 11, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # conv2
            nn.Conv2d(64, 192, kernel_size = 5),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # conv3
            nn.Conv2d(192, 384, kernel_size = 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace = True),
            # conv4
            nn.Conv2d(384, 256, kernel_size = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            # conv5
            nn.Conv2d(256, 256, kernel_size = 3),
            nn.BatchNorm2d(256))

        self.conv_reg_z = nn.Conv2d(256, 256 * 4 * self.anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(256, 256, 3)
        self.conv_cls_z = nn.Conv2d(256, 256 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(256, 256, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num*1, 1)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 256, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 256, k, k)

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
        #self.net = SiamRPN()
        self.model = SiameseAlexNet()
        self.model.eval()
        self.data_loader = TrainDataLoader(self.model, params)

        '''setup GPU device if available'''
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        if self.cuda:
            self.model.cuda()

        if net_path is not None:
            self.model.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))

        #self.net.eval()

        anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
        anchor_scales = np.array([8, ])
        anchor_num = len(anchor_scales) * len(anchor_ratios)

        self.window = np.tile(np.outer(np.hanning(19), np.hanning(19))[None, :],
                              [anchor_num, 1, 1]).flatten()

    def init(self, target_img, target_box):

        self.pos = np.array(
            [target_box[0] + target_box[2] / 2 - 1 / 2, target_box[1] + target_box[3] / 2 - 1 / 2])  # center x, center y, zero based
        # self.pos = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])  # same to original code
        self.target_sz = np.array([target_box[2], target_box[3]])  # width, height
        self.bbox = np.array([target_box[0] + target_box[2] / 2 - 1 / 2, target_box[1] + target_box[3] / 2 - 1 / 2, target_box[2], target_box[3]])
        # self.bbox = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]])  # same to original code
        self.origin_target_sz = np.array([target_box[2], target_box[3]])
        # get exemplar img
        self.img_mean = np.mean(target_img, axis=(0, 1))

        self.box = target_box
        target_imgX = target_img

        target_img = np.asarray(target_img)

        target_centor, target_size = util.x1y1_wh_to_xy_wh(target_box) # x1y1wh -> xywh # convert to bauding box centor

        self.state = dict()
        p = TrackerConfig()
        self.state['target_img_h'] = target_img.shape[0]
        self.state['target_img_w'] = target_img.shape[1]

        if ((target_size[0] * target_size[1]) / float(self.state['target_img_h'] * self.state['target_img_w'])) < 0.004:
            p.detection_img_size = 287  # small object big search region

        p.score_size = int((p.detection_img_size - p.template_img_size) / p.total_stride + 1)

        p.anchor = util.generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)
        #print('p.anchor', p.anchor)


        avg_chans = np.mean(target_img, axis=(0, 1))

        wc_z = target_size[0] + p.context_amount * sum(target_size)
        print('wc_z', wc_z)
        hc_z = target_size[1] + p.context_amount * sum(target_size)
        print('hc_z', hc_z)

        s_z = round(np.sqrt(wc_z * hc_z))

        # initialize the exemplar
        z_crop = util.get_subwindow_tracking(target_img, target_centor, p.template_img_size, s_z, avg_chans)

        self.ret = self.data_loader.get_template(target_imgX, self.box)



        '''z = Variable(z_crop.unsqueeze(0))
        self.kernel_reg, self.kernel_cls = self.net.learn(ret['template_tensor'])#.cuda())'''
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

        self.model.track_init(self.ret['template_tensor'])

    def update(self, im):
        p = self.state['p']
        ret = self.data_loader.__get__(im, self.box)
        self.detection_tensor = ret['detection_tensor']

        pred_score, pred_regression = self.model.track(self.detection_tensor)

        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        pred_offset = pred_regression.reshape(-1, 4,
                                              config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)

        scale_x = ret['detection_cropped_resized_ratio']


        #rout, cout = self.net.inference(self.detection_tensor, self.kernel_reg, self.kernel_cls)

        #pred_conf = cout.permute(0,2,3,1).reshape(1,-1, 2)

        #pred_offset = rout.permute(0,2,3,1).reshape(1,-1, 4)

        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = util.box_transform_inv(p.anchor, delta)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()

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

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - p.window_influence) + self.window * p.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * p.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, im.size[1])
        res_y = np.clip(target[1] + self.pos[1], 0, im.size[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, p.min_scale * self.origin_target_sz[0],
                        p.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, p.min_scale * self.origin_target_sz[1],
                        p.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])

        self.bbox = (   np.clip(bbox[0], 0, im.size[1]).astype(np.float64),
                        np.clip(bbox[1], 0, im.size[0]).astype(np.float64),
                        np.clip(bbox[2], 10, im.size[1]).astype(np.float64),
                        np.clip(bbox[3], 10, im.size[0]).astype(np.float64))

        return self.bbox
