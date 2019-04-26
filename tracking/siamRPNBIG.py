import cv2
import torch
import numpy as np
import torch.nn as nn
from util import util
import torch.nn.functional as F
from config import TrackerConfig
import torchvision.transforms as transforms
from custom_transforms import ToTensor
from config import config
from torch.autograd import Variable
from got10k.trackers import Tracker
from network import SiameseAlexNet
from data_loader import TrackerDataLoader
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
    def __init__(self, params, model_path = None, **kargs):
        super(TrackerSiamRPNBIG, self).__init__(name='SiamRPN', is_deterministic=True)

        self.model = SiameseAlexNet()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        checkpoint = torch.load(model_path, map_location = self.device)
        #print("1")
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path, map_location = self.device)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path, map_location = self.device))


        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

        valid_scope = 2 * config.valid_scope + 1
        self.anchors = util.generate_anchors(   config.total_stride,
                                                config.anchor_base_size,
                                                config.anchor_scales,
                                                config.anchor_ratios,
                                                valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

        self.data_loader = TrackerDataLoader()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):

        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        frame = np.asarray(frame)
        '''bbox[0] = bbox[0] + bbox[2]/2
        bbox[1] = bbox[1] + bbox[3]/2'''

        self.pos = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        #self.pos = np.array([bbox[0], bbox[1]])  # center x, center y, zero based

        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
        #self.bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = self.data_loader.get_exemplar_image(   frame,
                                                                    self.bbox,
                                                                    config.template_img_size,
                                                                    config.context_amount,
                                                                    self.img_mean)

        #cv2.imshow('exemplar_img', exemplar_img)
        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        if self.cuda:
            self.model.track_init(exemplar_img.cuda())
        else:
            self.model.track_init(exemplar_img)

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        frame = np.asarray(frame)

        instance_img, _, _, scale_x = self.data_loader.get_instance_image(  frame,
                                                                            self.bbox,
                                                                            config.template_img_size,
                                                                            config.detection_img_size,
                                                                            config.context_amount,
                                                                            self.img_mean)
        #cv2.imshow('instance_img', instance_img)

        instance_img = self.transforms(instance_img)[None, :, :, :]
        if self.cuda:
            pred_score, pred_regression = self.model.track(instance_img.cuda())
        else:
            pred_score, pred_regression = self.model.track(instance_img)

        pred_conf   = pred_score.reshape(-1, 2, config.size ).permute(0, 2, 1)
        pred_offset = pred_regression.reshape(-1, 4, config.size ).permute(0, 2, 1)

        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = util.box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()

        s_c = util.change(util.sz(box_pred[:, 2], box_pred[:, 3]) / (util.sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = util.change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])

        bbox = np.array([res_x, res_y, res_w, res_h])
        #print('bbox', bbox)
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))

        res_x = res_x - res_w/2 # x -> x1
        res_y = res_y - res_h/2 # y -> y1
        bbox = np.array([res_x, res_y, res_w, res_h])
        return bbox
