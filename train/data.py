# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import random
import numpy as np
import os.path as osp
from util import util
from config import config
from torch.utils.data import Dataset
from got10k.datasets import ImageNetVID, GOT10k
from torchvision import datasets, transforms, utils
from PIL import Image, ImageOps, ImageStat, ImageDraw

class TrainDataLoader(Dataset):
    def __init__(self, seq_dataset, name):

        self.max_inter        = config.max_inter
        self.sub_class_dir    = seq_dataset
        self.ret              = {}
        self.count            = 0
        self.name             = name
        self.anchors          = util.generate_anchors(  config.total_stride,
                                                        config.anchor_base_size,
                                                        config.anchor_scales,
                                                        config.anchor_ratios,
                                                        config.anchor_valid_scope) #centor

    def get_transform_for_train(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
        return transforms.Compose(transform_list)

    # tuple
    def _average(self):
        assert self.ret.__contains__('template_img_path'), 'no template_img_path'
        assert self.ret.__contains__('detection_img_path'),'no detection_img_path'
        template = Image.open(self.ret['template_img_path'])
        detection= Image.open(self.ret['detection_img_path'])

        mean_template = tuple(map(round, ImageStat.Stat(template).mean))
        mean_detection= tuple(map(round, ImageStat.Stat(detection).mean))
        self.ret['mean_template'] = mean_template
        self.ret['mean_detection']= mean_detection

    def _pick_img_pairs(self, index_of_subclass):

        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'

        video_name = self.sub_class_dir[index_of_subclass][0]
        video_num  = len(video_name)
        video_gt = self.sub_class_dir[index_of_subclass][1]
        #print('video_num', video_num)

        status = True
        while status:
            if self.max_inter >= video_num-1:
                self.max_inter = video_num//2

            '''template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num-1)

            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num-1)

            template_img_path, detection_img_path  = video_name[template_index], video_name[detection_index]

            template_gt  = video_gt[template_index]

            detection_gt = video_gt[detection_index]'''

            template_index = random.choice(range(video_num-2))

            detection_index= template_index + 2

            template_img_path, detection_img_path  = video_name[template_index], video_name[detection_index]

            template_gt  = video_gt[template_index]

            detection_gt = video_gt[detection_index]

            if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:
                status = False
            else:
                print('Warning : Encounter object missing, reinitializing ...')

        # load infomation of template and detection
        self.ret['template_img_path']      = template_img_path
        self.ret['detection_img_path']     = detection_img_path
        self.ret['template_target_x1y1wh'] = template_gt
        self.ret['detection_target_x1y1wh']= detection_gt
        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()
        self.ret['template_target_xywh']   = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh']  = np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        self.ret['anchors'] = self.anchors
        self._average()

    def _pad_crop_resize(self):
        template_img, detection_img = Image.open(self.ret['template_img_path']), Image.open(self.ret['detection_img_path'])

        w, h = template_img.size
        cx, cy, tw, th = self.ret['template_target_xywh']
        p = round((tw + th)/2, 2)
        template_square_size  = int(np.sqrt((tw + p)*(th + p))) #a
        detection_square_size = int(template_square_size * 2)   #A =2a

        # pad
        detection_lt_x, detection_lt_y = cx - detection_square_size//2, cy - detection_square_size//2
        detection_rb_x, detection_rb_y = cx + detection_square_size//2, cy + detection_square_size//2
        left   = -detection_lt_x if detection_lt_x < 0 else 0
        top    = -detection_lt_y if detection_lt_y < 0 else 0
        right  =  detection_rb_x - w if detection_rb_x > w else 0
        bottom =  detection_rb_y - h if detection_rb_y > h else 0
        padding = tuple(map(int, [left, top, right, bottom]))
        new_w, new_h = left + right + w, top + bottom + h

        # pad load
        self.ret['padding'] = padding
        self.ret['new_template_img_padding_size'] = (new_w, new_h)
        self.ret['new_template_img_padding']      = ImageOps.expand(template_img,  border=padding, fill=self.ret['mean_template'])
        self.ret['new_detection_img_padding']     = ImageOps.expand(detection_img, border=padding, fill=self.ret['mean_detection'])

        # crop
        tl = cx + left - template_square_size//2
        tt = cy + top  - template_square_size//2
        tr = new_w - tl - template_square_size
        tb = new_h - tt - template_square_size
        self.ret['template_cropped'] = ImageOps.crop(self.ret['new_template_img_padding'], (tl, tt, tr, tb))

        dl = np.clip(cx + left - detection_square_size//2, 0, new_w - detection_square_size)
        dt = np.clip(cy + top  - detection_square_size//2, 0, new_h - detection_square_size)
        dr = np.clip(new_w - dl - detection_square_size, 0, new_w - detection_square_size)
        db = np.clip(new_h - dt - detection_square_size, 0, new_h - detection_square_size )
        self.ret['detection_cropped']= ImageOps.crop(self.ret['new_detection_img_padding'], (dl, dt, dr, db))

        self.ret['detection_tlcords_of_original_image'] = (cx - detection_square_size//2 , cy - detection_square_size//2)
        self.ret['detection_tlcords_of_padding_image']  = (cx - detection_square_size//2 + left, cy - detection_square_size//2 + top)
        self.ret['detection_rbcords_of_padding_image']  = (cx + detection_square_size//2 + left, cy + detection_square_size//2 + top)

        # resize
        self.ret['template_cropped_resized'] = self.ret['template_cropped'].copy().resize((config.template_img_size, config.template_img_size))   #(127, 127)
        #self.ret['template_cropped_resized'].show()
        self.ret['detection_cropped_resized']= self.ret['detection_cropped'].copy().resize((config.detection_img_size, config.detection_img_size))#(271, 271)
        self.ret['template_cropprd_resized_ratio'] = round(config.template_img_size/template_square_size, 2)
        self.ret['detection_cropped_resized_ratio'] = round(config.detection_img_size/detection_square_size, 2)

        # compute target in detection, and then we will compute IOU
        # whether target in detection part
        x, y, w, h = self.ret['detection_target_xywh']
        self.ret['target_tlcords_of_padding_image'] = np.array([int(x+left-w//2), int(y+top-h//2)], dtype = np.float32)
        self.ret['target_rbcords_of_padding_image'] = np.array([int(x+left+w//2), int(y+top+h//2)], dtype = np.float32)


        ### use cords of padding to compute cords about detection
        ### modify cords because not all the object in the detection
        x11, y11 = self.ret['detection_tlcords_of_padding_image']
        x12, y12 = self.ret['detection_rbcords_of_padding_image']

        x21, y21 = self.ret['target_tlcords_of_padding_image']
        x22, y22 = self.ret['target_rbcords_of_padding_image']
        x1_of_d, y1_of_d, x3_of_d, y3_of_d = int(x21-x11), int(y21-y11), int(x22-x11), int(y22-y11)
        #print('x21-x11', x21-x11)
        #print('x21,x11', x21,x11)

        #print('x22-x11', x22-x11)

        x1 = np.clip(x1_of_d, 0, x12-x11).astype(np.float32)
        y1 = np.clip(y1_of_d, 0, y12-y11).astype(np.float32)

        x2 = np.clip(x3_of_d, 0, x12-x11).astype(np.float32)
        y2 = np.clip(y3_of_d, 0, y12-y11).astype(np.float32)
        #print('x2,x1', x2,x1)

        self.ret['target_in_detection_x1y1x2y2']=np.array([x1, y1, x2, y2], dtype = np.float32)


        cords_in_cropped_detection = np.array((x1, y1, x2, y2), dtype = np.float32)
        cords_in_cropped_resized_detection = (cords_in_cropped_detection * self.ret['detection_cropped_resized_ratio']).astype(np.int32)
        x1, y1, x2, y2 = cords_in_cropped_resized_detection
        cx, cy, w, h = (x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1
        #print('x2,x1', x2,x1)
        #print('x2-x1', x2-x1)


        self.ret['target_in_resized_detection_x1y1x2y2'] = np.array((x1, y1, x2, y2), dtype = np.int32)
        self.ret['target_in_resized_detection_xywh']     = np.array((cx, cy, w,  h) , dtype = np.int32)
        self.ret['area_target_in_resized_detection']     = w * h


    def _target(self):
        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round,
                                                             self.ret['target_in_resized_detection_xywh']))))

        #print(self.ret['target_in_resized_detection_xywh'])

        return regression_target, conf_target

    def compute_target(self, anchors, box):
        regression_target = self.box_transform(anchors, box)

        iou = self.compute_iou(anchors, box).flatten()
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

    def box_transform(self, anchors, gt_box):
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box

        target_x = (gt_cx - anchor_xctr) / anchor_w
        target_y = (gt_cy - anchor_yctr) / anchor_h
        target_w = np.log(gt_w / anchor_w)
        target_h = np.log(gt_h / anchor_h)
        regression_target = np.hstack((target_x, target_y, target_w, target_h))
        return regression_target

    def compute_iou(self, anchors, box):
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                               axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return iou

    def _tranform(self):
        """PIL to Tensor"""
        template_pil = self.ret['template_cropped_resized'].copy()
        detection_pil= self.ret['detection_cropped_resized'].copy()
        #pos_neg_diff = self.ret['pos_neg_diff'].copy()

        transform       = self.get_transform_for_train()
        template_tensor = transform(template_pil)
        detection_tensor= transform(detection_pil)

        self.ret['template_tensor']     = template_tensor#.unsqueeze(0)

        self.ret['detection_tensor']    = detection_tensor#.unsqueeze(0)

        #self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)

    def __getitem__(self, index):

        index = random.choice(range(len(self.sub_class_dir)))
        if self.name == 'GOT-10k':
            if index == 8627 or index == 8629 or index == 9057 or index == 9058:
                index += 1

        self._pick_img_pairs(index)
        self._pad_crop_resize()
        #self._generate_pos_neg_diff()
        self._tranform()
        regression_target, conf_target = self._target()
        self.count += 1
        '''detection = self.ret['detection_cropped_resized']

        draw = ImageDraw.Draw(detection)

        x, y, w, h = self.ret['target_in_resized_detection_xywh']
        x1, y1, x2, y2 = x-w//2, y-h//2, x+w//2, y+h//2
        draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')

        detection.show()
        '''
        return self.ret['template_tensor'], self.ret['detection_tensor'], regression_target, conf_target.astype(np.int64)
    def __len__(self):
        return config.train_epoch_size
