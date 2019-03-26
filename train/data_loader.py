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
    def __init__(self, feature_w, feature_h, params):
        #self.w      = feature_w
        #self.h      = feature_h
        #self.base   = 64                   # base size for anchor box
        #self.stride = 16                   # center point shift stride
        #self.scale  = [1/3, 1/2, 1, 2, 3]  # aspect ratio
        self.anchors = self.create_anchors(params)   # xywh
        self.eps     = 0.01

    def create_anchors(self, params):
        response_sz = (params['detection_img_size'] - params['template_img_size']) // params['stride'] + 1 #19

        anchor_num = len(params['ratios']) * len(params['scales'])
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = params['stride'] * params['stride']
        ind = 0
        for ratio in params['ratios']:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in params['scales']:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * params['stride']
        xs, ys = np.meshgrid(
            begin + params['stride'] * np.arange(response_sz),
            begin + params['stride'] * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)
        #print('anchors', anchors)
        #print('anchors.shape', anchors.shape)
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
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))           # box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))           # box1=[N,K,4]
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
    def __init__(self, img_dir_path, params, out_feature = 19, max_inter = 80):
        assert osp.isdir(img_dir_path), 'input img_dir_path error'
        self.anchor_generator = Anchor_ms(out_feature, out_feature, params)
        self.img_dir_path = img_dir_path # this is a root dir contain subclass
        self.max_inter = max_inter
        self.sub_class_dir = [sub_class_dir for sub_class_dir in os.listdir(img_dir_path) if os.path.isdir(os.path.join(img_dir_path, sub_class_dir))]
        #self.anchors = self.anchor_generator.gen_anchors() #centor
        self.anchors = self.anchor_generator.create_anchors(params) #centor
        self.ret = {}
        self.count = 0
        self.params = params


    def get_transform_for_train(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
        return transforms.Compose(transform_list)

    # tuple
    def _average(self):
        assert self.ret.__contains__('template_img_path'), 'no template path'
        assert self.ret.__contains__('detection_img_path'),'no detection path'
        template = Image.open(self.ret['template_img_path'])
        detection= Image.open(self.ret['detection_img_path'])

        mean_template = tuple(map(round, ImageStat.Stat(template).mean))
        mean_detection= tuple(map(round, ImageStat.Stat(detection).mean))
        self.ret['mean_template'] = mean_template
        self.ret['mean_detection']= mean_detection

    def _pick_img_pairs(self, index_of_subclass):
        # img_dir_path -> sub_class_dir_path -> template_img_path
        # use index_of_subclass to select a sub directory
        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'
        sub_class_dir_basename = self.sub_class_dir[index_of_subclass]
        sub_class_dir_path = os.path.join(self.img_dir_path, sub_class_dir_basename)
        sub_class_img_name = [img_name for img_name in os.listdir(sub_class_dir_path) if not img_name.find('.jpg') == -1]
        sub_class_img_name = sorted(sub_class_img_name)
        sub_class_img_num  = len(sub_class_img_name)
        sub_class_gt_name  = 'groundtruth.txt'

        # select template, detection
        # ++++++++++++++++++++++++++++ add break in sequeence [0,0,0,0] ++++++++++++++++++++++++++++++++++
        status = True
        while status:
            if self.max_inter >= sub_class_img_num-1:
                self.max_inter = sub_class_img_num//2

            template_index = np.clip(random.choice(range(0, max(1, sub_class_img_num - self.max_inter))), 0, sub_class_img_num-1)
            #print('template_index', template_index)
            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, sub_class_img_num-1)
            #print('detection_index', detection_index)


            template_name, detection_name  = sub_class_img_name[template_index], sub_class_img_name[detection_index]
            template_img_path, detection_img_path = osp.join(sub_class_dir_path, template_name), osp.join(sub_class_dir_path, detection_name)
            gt_path = osp.join(sub_class_dir_path, sub_class_gt_name)
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            cords_of_template_abs  = [abs(int(float(i))) for i in lines[template_index].strip('\n').split(',')[:4]]
            cords_of_detection_abs = [abs(int(float(i))) for i in lines[detection_index].strip('\n').split(',')[:4]]

            if cords_of_template_abs[2]*cords_of_template_abs[3]*cords_of_detection_abs[2]*cords_of_detection_abs[3] != 0:
                status = False
            else:
                print('Warning : Encounter object missing, reinitializing ...')

        # load infomation of template and detection
        self.ret['template_img_path']      = template_img_path
        self.ret['detection_img_path']     = detection_img_path
        self.ret['template_target_x1y1wh'] = [int(float(i)) for i in lines[template_index].strip('\n').split(',')[:4]]
        self.ret['detection_target_x1y1wh']= [int(float(i)) for i in lines[detection_index].strip('\n').split(',')[:4]]
        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()
        self.ret['template_target_xywh'] = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh']= np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        self.ret['anchors'] = self.anchors
        self._average()

    def _pad_crop_resize(self):
        #print('self.ret[template_img_path]', self.ret['template_img_path'])
        #print('self.ret[detection_img_path]', self.ret['detection_img_path'])
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
        self.ret['new_template_img_padding'] = ImageOps.expand(template_img,  border=padding, fill=self.ret['mean_template'])
        self.ret['new_detection_img_padding']= ImageOps.expand(detection_img, border=padding, fill=self.ret['mean_detection'])

        # crop
        #print('template_square_size', template_square_size)
        tl = cx + left - template_square_size//2
        tt = cy + top  - template_square_size//2
        tr = new_w - tl - template_square_size
        tb = new_h - tt - template_square_size
        #print('(tl, tt, tr, tb)', (tl, tt, tr, tb))
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
        self.ret['template_cropped_resized'] = self.ret['template_cropped'].copy().resize((self.params["template_img_size"], self.params["template_img_size"]))
        self.ret['detection_cropped_resized']= self.ret['detection_cropped'].copy().resize((self.params["detection_img_size"], self.params["detection_img_size"]))
        self.ret['template_cropprd_resized_ratio'] = round(self.params["template_img_size"]/template_square_size, 2)
        self.ret['detection_cropped_resized_ratio'] = round(self.params["detection_img_size"]/detection_square_size, 2)

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
        #template_pil.show()
        detection_pil= self.ret['detection_cropped_resized'].copy()
        #detection_pil.show()
        pos_neg_diff = self.ret['pos_neg_diff'].copy()

        transform = self.get_transform_for_train()
        template_tensor = transform(template_pil)
        detection_tensor= transform(detection_pil)
        self.ret['template_tensor'] = template_tensor.unsqueeze(0)
        self.ret['detection_tensor']= detection_tensor.unsqueeze(0)
        self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)


    def __get__(self, index):

        self._pick_img_pairs(index)
        self._pad_crop_resize()
        self._generate_pos_neg_diff()
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
