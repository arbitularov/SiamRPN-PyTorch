# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
from torchvision import datasets, transforms, utils

class TrackerDataLoader(object):

    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, img_mean=None):

        cx, cy, w, h = bbox  # float type
        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
        scale_z = size_z / s_z

        s_x = s_z * size_x / size_z
        instance_img, scale_x = self.crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
        w_x = w * scale_x
        h_x = h * scale_x
        # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
        # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
        # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
        # cv2.imwrite('1.jpg', frame)
        return instance_img, w_x, h_x, scale_x

    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        cx, cy, w, h = bbox
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = size_z / s_z
        exemplar_img, _ = self.crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
        return exemplar_img, scale_z, s_z

    def crop_and_pad(self, img, cx, cy, model_sz, original_sz, img_mean=None):

        def round_up(value):
            # 替换内置round函数,实现保留2位小数的精确四舍五入
            return round(value + 1e-6 + 1000) - 1000
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2
        ymax = ymin + original_sz - 1

        left = int(round_up(max(0., -xmin)))
        top = int(round_up(max(0., -ymin)))
        right = int(round_up(max(0., xmax - im_w + 1)))
        bottom = int(round_up(max(0., ymax - im_h + 1)))

        xmin = int(round_up(xmin + left))
        xmax = int(round_up(xmax + left))
        ymin = int(round_up(ymin + top))
        ymax = int(round_up(ymax + top))
        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]
        return im_patch, scale

    def box_transform_inv(self, anchors, offset):
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        offset_x, offset_y, offset_w, offset_h = offset[:, :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:],

        box_cx = anchor_w * offset_x + anchor_xctr
        box_cy = anchor_h * offset_y + anchor_yctr
        box_w = anchor_w * np.exp(offset_w)
        box_h = anchor_h * np.exp(offset_h)
        box = np.hstack([box_cx, box_cy, box_w, box_h])
        return box

    def generate_anchors(self, total_stride, base_size, scales, ratios, score_size):
        anchor_num = len(ratios) * len(scales)
        anchor = np.zeros((anchor_num, 4), dtype=np.float32)
        size = base_size * base_size
        count = 0
        for ratio in ratios:
            # ws = int(np.sqrt(size * 1.0 / ratio))
            ws = int(np.sqrt(size / ratio))
            hs = int(ws * ratio)
            for scale in scales:
                wws = ws * scale
                hhs = hs * scale
                anchor[count, 0] = 0
                anchor[count, 1] = 0
                anchor[count, 2] = wws
                anchor[count, 3] = hhs
                count += 1

        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        # (5,4x225) to (225x5,4)
        ori = - (score_size // 2) * total_stride
        # the left displacement
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        # (15,15)
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        # (15,15) to (225,1) to (5,225) to (225x5,1)
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor
