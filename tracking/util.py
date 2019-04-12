import cv2
import torch
import numpy as np

class Util(object):

    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, img_mean=None):

        cx, cy, w, h = bbox  # float type
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
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

    def get_subwindow_tracking(self, im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):

        # im (720, 1280, 3)
        # pos [406.  377.5]
        # model_sz 127
        # original_sz 768.0
        # avg_chans [115.18894748 111.79296549 109.10407878]

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz # original_sz 768.0
        im_sz = im.shape # im (720, 1280, 3)
        c = (original_sz+1) / 2 # 384.5
        context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        # zzp: a more easy speed version
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original

        cv2.imshow('foto', im_patch)

        def im_to_torch(img):

            def to_torch(ndarray):
                if type(ndarray).__module__ == 'numpy':
                    return torch.from_numpy(ndarray)
                elif not torch.is_tensor(ndarray):
                    raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
                return ndarray
            img = np.transpose(img, (2, 0, 1))  # C*H*W
            img = to_torch(img).float()
            return img

        return im_to_torch(im_patch) if out_mode in 'torch' else im_patch

    def cxy_wh_2_rect(self, pos, sz):
        return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index

    def x1y1_wh_to_xy_wh(self, rect):
        return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index

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

util = Util()
