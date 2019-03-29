# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
from torchvision import datasets, transforms, utils
from PIL import Image, ImageOps, ImageStat, ImageDraw

class Anchor_ms(object):
    def __init__(self, feature_w, feature_h):
        self.w      = feature_w
        self.h      = feature_h
        self.base   = 64                   # base size for anchor box
        self.stride = 15                   # center point shift stride
        self.scale  = [1/3, 1/2, 1, 2, 3]  # aspect ratio
        self.anchors= self.gen_anchors()   # xywh
        self.eps    = 0.01

    def gen_single_anchor(self):
        scale = np.array(self.scale, dtype = np.float32)
        s = self.base * self.base
        w, h = np.sqrt(s/scale), np.sqrt(s*scale)
        c_x, c_y = (self.stride-1)//2, (self.stride-1)//2
        anchor = np.vstack([c_x*np.ones_like(scale, dtype=np.float32), c_y*np.ones_like(scale, dtype=np.float32), w, h]).transpose()
        anchor = self.center_to_corner(anchor)
        return anchor

    def gen_anchors(self):
        anchor=self.gen_single_anchor()
        k = anchor.shape[0]
        delta_x, delta_y = [x*self.stride for x in range(self.w)], [y*self.stride for y in range(self.h)]
        shift_x, shift_y = np.meshgrid(delta_x, delta_y)
        shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
        a = shifts.shape[0]
        anchors = (anchor.reshape((1,k,4))+shifts.reshape((a,1,4))).reshape((a*k, 4)) # corner format
        anchors = self.corner_to_center(anchors)
        return anchors

    # float
    def diff_anchor_gt(self, gt):
        eps = self.eps
        anchors, gt = self.anchors.copy(), gt.copy()
        diff = np.zeros_like(anchors, dtype = np.float32)
        diff[:,0] = (gt[0] - anchors[:,0])/(anchors[:,2] + eps)
        diff[:,1] = (gt[1] - anchors[:,1])/(anchors[:,3] + eps)
        diff[:,2] = np.log((gt[2] + eps)/(anchors[:,2] + eps))
        diff[:,3] = np.log((gt[3] + eps)/(anchors[:,3] + eps))
        return diff

    # float
    def center_to_corner(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[:,0]=box[:,0]-(box[:,2]-1)/2
        box_[:,1]=box[:,1]-(box[:,3]-1)/2
        box_[:,2]=box[:,0]+(box[:,2]-1)/2
        box_[:,3]=box[:,1]+(box[:,3]-1)/2
        box_ = box_.astype(np.float32)
        return box_

    # float
    def corner_to_center(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[:,0]=box[:,0]+(box[:,2]-box[:,0])/2
        box_[:,1]=box[:,1]+(box[:,3]-box[:,1])/2
        box_[:,2]=(box[:,2]-box[:,0])
        box_[:,3]=(box[:,3]-box[:,1])
        box_ = box_.astype(np.float32)
        return box_

    def pos_neg_anchor(self, gt, pos_num=16, neg_num=48, threshold_pos=0.5, threshold_neg=0.1):
        gt = gt.copy()
        gt_corner = self.center_to_corner(np.array(gt, dtype = np.float32).reshape(1, 4))
        an_corner = self.center_to_corner(np.array(self.anchors, dtype = np.float32))
        iou_value = self.iou(an_corner, gt_corner).reshape(-1) #(1445)
        max_iou   = max(iou_value)
        pos, neg  = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # pos
        pos_cand = np.argsort(iou_value)[::-1][:30]
        pos_index = np.random.choice(pos_cand, pos_num, replace = False)
        if max_iou > threshold_pos:
            pos[pos_index] = 1

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]
        neg_ind = np.random.choice(neg_cand, neg_num, replace = False)
        neg[neg_ind] = 1

        return pos, neg

    def iou(self,box1,box2):
        box1, box2 = box1.copy(), box2.copy()
        N=box1.shape[0]
        K=box2.shape[0]
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))#box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))#box1=[N,K,4]
        x_max=np.max(np.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=np.min(np.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=np.max(np.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=np.min(np.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        tb[np.where(tb<0)]=0
        lr[np.where(lr<0)]=0
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square

class TrainDataLoader(object):
    def __init__(self, model, params, out_feature = 19):

        self.anchor_generator = Anchor_ms(out_feature, out_feature)
        self.anchors = self.anchor_generator.gen_anchors() #centor
        self.ret = {}
        self.count = 0
        self.params = params
        self.model = model

    def get_template(self, image, box):

        self.ret['template_cropped_resized'] = self.template_crop_resize(image, box)
        #self.ret['template_cropped_resized'].show()
        transform = self.get_transform_for_train()
        template_tensor = transform(self.ret['template_cropped_resized'])
        self.ret['template_tensor'] = template_tensor.unsqueeze(0)
        self.ret['anchors'] = self.anchors

        return self.ret

    def generate_path(self, image, box):

        self.ret['detection_cropped_resized'], self.ret['target_in_resized_detection_xywh'] = self.detection_crop_resize(image, box)
        #self.ret['detection_cropped_resized'].show()

    def template_crop_resize(self, image, box):
        #template_img = Image.open(template_path)
        template_img = image

        w, h = template_img.size
        t1 = box
        cx, cy, tw, th = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)

        a = (tw + th)/2
        p = round(a*0.5, 2)
        template_square_size  = int(np.sqrt((tw + p)*(th + p)))
        detection_square_size = int(template_square_size * 2)

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
        mean_template = tuple(map(round, ImageStat.Stat(template_img).mean))
        template_img_padding = ImageOps.expand(template_img,  border=padding, fill=mean_template)

        # crop
        tl = cx + left - template_square_size//2
        tt = cy + top  - template_square_size//2
        tr = new_w - tl - template_square_size
        tb = new_h - tt - template_square_size
        template_cropped = ImageOps.crop(template_img_padding, (tl, tt, tr, tb))

        # resize
        template_cropped_resized = template_cropped.resize((127, 127))

        template_cropprd_resized_ratio = round(127/template_square_size, 2)

        return template_cropped_resized

    def detection_crop_resize(self, image, box):
        detection_img = image

        w, h = detection_img.size
        t1 = box
        cx, cy, tw, th = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)

        p = round((tw + th)/2, 2)

        template_square_size  = int(np.sqrt((tw + p)*(th + p)))
        detection_square_size = int(template_square_size * 2)

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
        mean_template = tuple(map(round, ImageStat.Stat(detection_img).mean))
        detection_img_padding = ImageOps.expand(detection_img, border=padding, fill=mean_template)
        #detection_img_padding.show()

        # crop
        dl = np.clip(cx + left - detection_square_size//2, 0, new_w - detection_square_size)
        dt = np.clip(cy + top  - detection_square_size//2, 0, new_h - detection_square_size)
        dr = np.clip(new_w - dl - detection_square_size, 0, new_w - detection_square_size)
        db = np.clip(new_h - dt - detection_square_size, 0, new_h - detection_square_size )

        detection_cropped = ImageOps.crop(detection_img_padding, (dl, dt, dr, db))
        #self.ret['detection_cropped'].show()

        detection_tlcords_of_padding_image  = (cx - detection_square_size//2 + left, cy - detection_square_size//2 + top)
        detection_rbcords_of_padding_image  = (cx + detection_square_size//2 + left, cy + detection_square_size//2 + top)

        # resize
        detection_cropped_resized = detection_cropped.resize((271, 271))
        #detection_cropped_resized.show()

        detection_cropped_resized_ratio = round(271/detection_square_size, 2)

        # compute target in detection, and then we will compute IOU
        # whether target in detection part
        x, y, w, h = cx, cy, tw, th
        target_tlcords_of_padding_image = np.array([int(x+left-w//2), int(y+top-h//2)], dtype = np.float32)
        target_rbcords_of_padding_image = np.array([int(x+left+w//2), int(y+top+h//2)], dtype = np.float32)

        ### use cords of padding to compute cords about detection
        ### modify cords because not all the object in the detection
        x11, y11 = detection_tlcords_of_padding_image[0], detection_tlcords_of_padding_image[1]
        x12, y12 = detection_rbcords_of_padding_image[0], detection_rbcords_of_padding_image[1]
        x21, y21 = target_tlcords_of_padding_image
        x22, y22 = target_rbcords_of_padding_image
        x1_of_d, y1_of_d, x3_of_d, y3_of_d = int(x21-x11), int(y21-y11), int(x22-x11), int(y22-y11)
        x1 = np.clip(x1_of_d, 0, x12-x11).astype(np.float32)
        y1 = np.clip(y1_of_d, 0, y12-y11).astype(np.float32)
        x2 = np.clip(x3_of_d, 0, x12-x11).astype(np.float32)
        y2 = np.clip(y3_of_d, 0, y12-y11).astype(np.float32)
        target_in_detection_x1y1x2y2 = np.array([x1, y1, x2, y2], dtype = np.float32)

        cords_in_cropped_detection = np.array((x1, y1, x2, y2), dtype = np.float32)
        cords_in_cropped_resized_detection = (cords_in_cropped_detection * detection_cropped_resized_ratio).astype(np.int32)
        x1, y1, x2, y2 = cords_in_cropped_resized_detection
        cx, cy, w, h = (x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1
        target_in_resized_detection_x1y1x2y2 = np.array((x1, y1, x2, y2), dtype = np.int32)
        target_in_resized_detection_xywh     = np.array((cx, cy, w,  h) , dtype = np.int32)
        area_target_in_resized_detection     = w * h

        return detection_cropped_resized, target_in_resized_detection_xywh

    def get_transform_for_train(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
        return transforms.Compose(transform_list)

    def tranform(self):
        """PIL to Tensor"""
        detection_pil= self.ret['detection_cropped_resized'].copy()

        transform = self.get_transform_for_train()
        detection_tensor= transform(detection_pil)
        self.ret['detection_tensor']= detection_tensor.unsqueeze(0)

    def __get__(self, image, box):
        self.generate_path(image, box)
        self.tranform()
        self.count += 1
        return self.ret
