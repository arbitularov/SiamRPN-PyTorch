# -*- coding: utf-8 -*-

import cv2
import torch
import random
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
    def __init__(self, seq_dataset, params, out_feature = 19):

        self.anchor_generator = Anchor_ms(out_feature, out_feature)
        self.anchors = self.anchor_generator.gen_anchors() #centor
        self.ret = {}
        self.count = 0
        self.params = params

        self.video = 0
        self.frame = 0
        self.array = []

        self.seq_dataset = seq_dataset

    def crop_and_resize(self):

        index_video = range(len(self.seq_dataset))

        index_frame = len(self.seq_dataset[self.video][0])

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

            self.template_frame_resized = self._crop_and_resize(template_frame, self.center, self.z_sz, self.params['template_img_size'], self.avg_color)

            self.detection_frame_resized = self._crop_and_resize(detection_frame, self.center, self.x_sz, self.params['detection_img_size'], self.avg_color)
            img = self.detection_frame_resized.copy()


            self.template_frame_tensor = torch.from_numpy(self.template_frame_resized).permute([2, 0, 1]).unsqueeze(0).float()
            self.ret['template_frame_tensor'] = self.template_frame_tensor

        else:
            self.detection_frame = self.array[self.frame+1]
            self.detection_box   = self.seq_dataset[self.video][1][self.frame+1]

            self.frame += 1

            self.detection_frame = np.asarray(self.detection_frame)

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

            self.detection_frame_resized = self._crop_and_resize(self.detection_frame, self.center, self.x_sz, self.params['detection_img_size'], self.avg_color)
            if self.frame+1 == index_frame:
                self.video = random.choice(index_video)
                self.frame = 0

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        '''convert box to corners (0-indexed)'''
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        '''pad image if necessary'''
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))

        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        '''crop image patch'''
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]
        #cv2.imshow('image2',patch)

        '''resize to out_size'''
        patch = cv2.resize(patch, (out_size, out_size))
        cv2.imshow('image3',patch)

        return patch

    def _pad_crop_resize(self):

        t1, t2 = self.template_box.copy(), self.detection_box.copy()
        print('t1, t2', t1, t2)
        self.ret['template_target_xywh'] = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32) # center for template bouding box
        self.ret['detection_target_xywh']= np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32) # center for detection bouding box
        self.ret['anchors'] = self.anchors

        w, h = self.template_frame.shape[1], self.template_frame.shape[0]
        print('w, h', w, h)
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

        self.ret['detection_tlcords_of_padding_image']  = (cx - detection_square_size//2 + left, cy - detection_square_size//2 + top)
        self.ret['detection_rbcords_of_padding_image']  = (cx + detection_square_size//2 + left, cy + detection_square_size//2 + top)

        # resize

        self.ret['template_cropped_resized'] = self.template_frame_resized

        self.ret['detection_cropped_resized']= self.detection_frame_resized

        self.ret['template_cropprd_resized_ratio'] = round(127/template_square_size, 2)
        self.ret['detection_cropped_resized_ratio'] = round(271/detection_square_size, 2)


        '''compute target in detection, and then we will compute IOU'''
        '''whether target in detection part'''

        x, y, w, h = self.ret['detection_target_xywh']
        self.ret['target_tlcords_of_padding_image'] = np.array([int(x+left-w//2), int(y+top-h//2)], dtype = np.float32)
        self.ret['target_rbcords_of_padding_image'] = np.array([int(x+left+w//2), int(y+top+h//2)], dtype = np.float32)


        '''use cords of padding to compute cords about detection'''
        '''modify cords because not all the object in the detection'''

        x11, y11 = self.ret['detection_tlcords_of_padding_image']
        x12, y12 = self.ret['detection_rbcords_of_padding_image']
        x21, y21 = self.ret['target_tlcords_of_padding_image']
        x22, y22 = self.ret['target_rbcords_of_padding_image']

        x1_of_d, y1_of_d, x3_of_d, y3_of_d = int(x21-x11), int(y21-y11), int(x22-x11), int(y22-y11)

        x1 = np.clip(x1_of_d, 0, x12-x11).astype(np.float32)
        y1 = np.clip(y1_of_d, 0, y12-y11).astype(np.float32)
        x2 = np.clip(x3_of_d, 0, x12-x11).astype(np.float32)
        y2 = np.clip(y3_of_d, 0, y12-y11).astype(np.float32)
        self.ret['target_in_detection_x1y1x2y2']=np.array([x1, y1, x2, y2], dtype = np.float32)

        cords_in_cropped_detection = np.array((x1, y1, x2, y2), dtype = np.float32)
        cords_in_cropped_resized_detection = (cords_in_cropped_detection * self.ret['detection_cropped_resized_ratio']).astype(np.int32)
        x1, y1, x2, y2 = cords_in_cropped_resized_detection
        cx, cy, w, h = (x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1

        self.ret['target_in_resized_detection_x1y1x2y2'] = np.array((x1, y1, x2, y2), dtype = np.int32)
        self.ret['target_in_resized_detection_xywh']     = np.array((cx, cy, w,  h) , dtype = np.int32)
        self.ret['area_target_in_resized_detection'] = w * h

    def _generate_pos_neg_diff(self):
        gt_box_in_detection = self.ret['target_in_resized_detection_xywh'].copy()
        print('self.ret[\'target_in_resized_detection_xywh\']', self.ret['target_in_resized_detection_xywh'])
        pos, neg = self.anchor_generator.pos_neg_anchor(gt_box_in_detection)
        diff     = self.anchor_generator.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1,1)), diff.reshape((-1, 4))
        class_target = np.array([-100.] * self.anchors.shape[0], np.int32)

        '''pos'''
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        self.ret['pos_anchors'] = np.array(self.ret['anchors'][pos_index, :], dtype=np.int32) if not pos_num == 0 else None
        if pos_num > 0:
            class_target[pos_index] = 1

        '''neg'''
        neg_index = np.where(neg == 1)[0]
        neg_num = len(neg_index)
        class_target[neg_index] = 0

        class_logits = class_target.reshape(-1, 1)
        pos_neg_diff = np.hstack((class_logits, diff))
        self.ret['pos_neg_diff'] = pos_neg_diff
        return pos_neg_diff

    def _tranform(self):

        pos_neg_diff = self.ret['pos_neg_diff'].copy()

        self.detection_frame_tensor = torch.from_numpy(self.detection_frame_resized).permute([2, 0, 1]).unsqueeze(0).float()
        self.ret['detection_frame_tensor'] = self.detection_frame_tensor
        self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)

    def __get__(self):
        self.crop_and_resize()
        self._pad_crop_resize()
        self._generate_pos_neg_diff()
        self._tranform()
        self.count += 1
        return self.ret

    def __len__(self):
        return 0

if __name__ == '__main__':
    '''we will do a test for dataloader'''
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
