import sys
import os
import os.path as osp
import time
import cv2
import torch
import random
from PIL import Image, ImageOps, ImageStat, ImageDraw
from torchvision import datasets, transforms, utils
import numpy as np


class Anchor_ms(object):
    """
    stable version for anchor generator
    """
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
    def __init__(self, image, box, out_feature = 17, max_inter = 80):
        self.image = image
        self.box = box
        self.anchor_generator = Anchor_ms(out_feature, out_feature)
        self.max_inter = max_inter
        self.anchors = self.anchor_generator.gen_anchors() #centor
        self.ret = {}
        self.count = 0

    def get_transform_for_train(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
        return transforms.Compose(transform_list)

    # tuple
    def _average(self, detection):
        #assert self.ret.__contains__('template_img_path'), 'no template path'
        #assert self.ret.__contains__('detection_img_path'),'no detection path'
        #template = Image.open(self.ret['template_img_path'])
        #detection= Image.open(self.ret['detection_img_path'])

        mean_template = tuple(map(round, ImageStat.Stat(self.image).mean))
        mean_detection= tuple(map(round, ImageStat.Stat(detection).mean))
        self.ret['mean_template'] = mean_template
        self.ret['mean_detection']= mean_detection

    def _pick_img_pairs(self, detection):
        # img_dir_path -> sub_class_dir_path -> template_img_path
        # use index_of_subclass to select a sub directory
        self.ret['anchors'] = self.anchors
        self._average(detection)

    def _pad_crop_resize(self, detection_img):

        template_img, detection_img = self.image, detection_img

        w, h = template_img.size
        cx, cy, tw, th = self.box[0], self.box[1], self.box[2], self.box[3]
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
        self.ret['new_template_img_padding'] = ImageOps.expand(template_img,  border=padding, fill=self.ret['mean_template'])
        self.ret['new_detection_img_padding']= ImageOps.expand(detection_img, border=padding, fill=self.ret['mean_detection'])

        # crop
        tl = cx + left - template_square_size//2
        tt = cy + top  - template_square_size//2
        tr = new_w - tl - template_square_size
        tb = new_h - tt - template_square_size
        self.ret['template_cropped'] = ImageOps.crop(self.ret['new_template_img_padding'].copy(), (tl, tt, tr, tb))

        dl = np.clip(cx + left - detection_square_size//2, 0, new_w - detection_square_size)
        dt = np.clip(cy + top  - detection_square_size//2, 0, new_h - detection_square_size)
        dr = np.clip(new_w - dl - detection_square_size, 0, new_w - detection_square_size)
        db = np.clip(new_h - dt - detection_square_size, 0, new_h - detection_square_size )
        self.ret['detection_cropped']= ImageOps.crop(self.ret['new_detection_img_padding'].copy(), (dl, dt, dr, db))

        self.ret['detection_tlcords_of_original_image'] = (cx - detection_square_size//2 , cy - detection_square_size//2)
        self.ret['detection_tlcords_of_padding_image']  = (cx - detection_square_size//2 + left, cy - detection_square_size//2 + top)
        self.ret['detection_rbcords_of_padding_image']  = (cx + detection_square_size//2 + left, cy + detection_square_size//2 + top)

        # resize
        self.ret['template_cropped_resized'] = self.ret['template_cropped'].copy().resize((127, 127))
        self.ret['detection_cropped_resized']= self.ret['detection_cropped'].copy().resize((256, 256))
        self.ret['template_cropprd_resized_ratio'] = round(127/template_square_size, 2)
        self.ret['detection_cropped_resized_ratio'] = round(256/detection_square_size, 2)



    def _generate_pos_neg_diff(self):
        gt_box_in_detection = self.ret['target_in_resized_detection_xywh'].copy()
        pos, neg = self.anchor_generator.pos_neg_anchor(gt_box_in_detection)
        diff     = self.anchor_generator.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1,1)), diff.reshape((-1, 4))
        class_target = np.array([-100.] * self.anchors.shape[0], np.int32)

        # pos
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        self.ret['pos_anchors'] = np.array(self.ret['anchors'][pos_index, :], dtype=np.int32) if not pos_num == 0 else None
        if pos_num > 0:
            class_target[pos_index] = 1

        # neg
        neg_index = np.where(neg == 1)[0]
        neg_num = len(neg_index)
        class_target[neg_index] = 0

        # draw pos and neg anchor box


        class_logits = class_target.reshape(-1, 1)
        pos_neg_diff = np.hstack((class_logits, diff))
        self.ret['pos_neg_diff'] = pos_neg_diff
        return pos_neg_diff

    def _tranform(self):
        """PIL to Tensor"""
        template_pil = self.ret['template_cropped_resized'].copy()
        detection_pil= self.ret['detection_cropped_resized'].copy()
        #pos_neg_diff = self.ret['pos_neg_diff'].copy()

        transform = self.get_transform_for_train()
        template_tensor = transform(template_pil)
        detection_tensor= transform(detection_pil)
        self.ret['template_tensor'] = template_tensor.unsqueeze(0)
        self.ret['detection_tensor']= detection_tensor.unsqueeze(0)
        #self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)


    def __get__(self, detection):
        self._pick_img_pairs(detection)
        self._pad_crop_resize(detection)
        #self._generate_pos_neg_diff()
        self._tranform()
        self.count += 1
        return self.ret

    def __len__(self):
        return len(self.sub_class_dir)

if __name__ == '__main__':
    # we will do a test for dataloader
    loader = TrainDataLoader('/home/song/srpn/dataset/simple_vot13', check = True)
    #print(loader.__len__())
    index_list = range(loader.__len__())
    for i in range(1000):
        ret = loader.__get__(random.choice(index_list))
        label = ret['pos_neg_diff'][:, 0].reshape(-1)
        pos_index = list(np.where(label == 1)[0])
        pos_num = len(pos_index)
        print(pos_index)
        print(pos_num)
        if pos_num != 0 and pos_num != 16:
            print(pos_num)
            sys.exit(0)
        print(i)
