import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import cv2

class Util(object):

    def add_box_img(self, img, boxes, color=(0, 255, 0), x = 1, y = 1):
        # boxes (x,y,w,h)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        img = img.copy()
        img_ctx = (img.shape[1] - 1) / 2
        img_cty = (img.shape[0] - 1) / 2
        for box in boxes:
            point_1 = [img_ctx - box[2] / 2 + (box[0]/x) + 0.5, img_cty - box[3] / 2 + (box[1]/y) + 0.5]
            point_2 = [img_ctx + box[2] / 2 + (box[0]/x) - 0.5, img_cty + box[3] / 2 + (box[1]/y) - 0.5]
            point_1[0] = np.clip(point_1[0], 0, img.shape[1])
            point_2[0] = np.clip(point_2[0], 0, img.shape[1])
            point_1[1] = np.clip(point_1[1], 0, img.shape[0])
            point_2[1] = np.clip(point_2[1], 0, img.shape[0])
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        return img

    def get_topk_box(self, cls_score, pred_regression, anchors, topk=10):
        # anchors xc,yc,w,h
        regress_offset = pred_regression.cpu().detach().numpy()

        scores, index = torch.topk(cls_score, topk, )
        index = index.view(-1).cpu().detach().numpy()

        topk_offset = regress_offset[index, :]
        anchors = anchors[index, :]
        pred_box = self.box_transform_inv(anchors, topk_offset)
        return pred_box

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

    def data_split(self, seq_datasetVID, seq_datasetGOT):
        seq_dataset = []
        for i in seq_datasetVID:
            seq_dataset.append(i)

        for i, data in enumerate(seq_datasetGOT):
            seq_dataset.append(data)
            if i >= 8600:
                break
        return seq_dataset

    def generate_anchors(self, total_stride, base_size, scales, ratios, score_size):
        anchor_num = len(ratios) * len(scales) # 5
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
        ori = - (score_size // 2) * total_stride
        # the left displacement
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])

        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    # freeze layers
    def freeze_layers(self, model):
        for layer in model.featureExtract[:10]:
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.Conv2d):
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.MaxPool2d):
                continue
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                raise KeyError('error in fixing former 3 layers')

    def experiment_name_dir(self, experiment_name):
        experiment_name_dir = 'experiments/{}'.format(experiment_name)
        if experiment_name == 'default':
            print('You are using "default" experiment, my advice to you is: Copy "default" change folder name and change settings in file "parameters.json"')
        else:
            print('You are using "{}" experiment'.format(experiment_name))
        return experiment_name_dir

    def adjust_learning_rate(self, optimizer, decay=0.1):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    def nms(self, bboxes, scores, num, threshold=0.7):
        print('scores',  scores)
        sort_index = np.argsort(scores)[::-1]
        print('sort_index', sort_index)
        sort_boxes = bboxes[sort_index]
        selected_bbox = [sort_boxes[0]]
        selected_index = [sort_index[0]]
        for i, bbox in enumerate(sort_boxes):
            iou = compute_iou(selected_bbox, bbox)
            print(iou, bbox, selected_bbox)
            if np.max(iou) < threshold:
                selected_bbox.append(bbox)
                selected_index.append(sort_index[i])
                if len(selected_bbox) >= num:
                    break
        return selected_index

util = Util()

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SavePlot(object):
    def __init__(self,  exp_name_dir,
                        name = 'plot',
                        title  = 'Siamese RPN',
                        ylabel = 'loss',
                        xlabel = 'epoch',
                        show   = False):

        self.step = 0
        self.exp_name_dir = exp_name_dir
        self.steps_array  = []
        self.train_array  = []
        self.val_array    = []
        self.name   = name
        self.title  = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.show   = show

        self.plot(  self.exp_name_dir,
                    self.steps_array,
                    self.train_array,
                    self.val_array,
                    self.name,
                    self.title,
                    self.ylabel,
                    self.xlabel,
                    self.show)

        self.plt.legend()

    def update(self, train,
                     val,
                     train_label = 'train loss',
                     val_label   = 'val loss',
                     count_step=1):

        self.step += count_step
        self.steps_array.append(self.step)
        self.train_array.append(train)
        self.val_array.append(val)

        self.plot(exp_name_dir = self.exp_name_dir,
                        step   = self.steps_array,
                        train  = self.train_array,
                        val    = self.val_array,
                        name   = self.name,
                        title  = self.title,
                        ylabel = self.ylabel,
                        xlabel = self.xlabel,
                        show   = self.show,
                        train_label = train_label,
                        val_label   = val_label)

    def plot(self,  exp_name_dir,
                    step,
                    train,
                    val,
                    name,
                    title,
                    ylabel,
                    xlabel,
                    show,
                    train_label = 'train loss',
                    val_label   = 'val loss'):
        self.plt  = plt
        self.plt.plot(step, train, 'r', label = train_label, color = 'red')
        self.plt.plot(step, val, 'r', label = val_label, color='black')

        self.plt.title(title)
        self.plt.ylabel(ylabel)
        self.plt.xlabel(xlabel)

        '''save plot'''
        self.plt.savefig("{}/{}.png".format(exp_name_dir, name))
        if show:
            self.plt.show()
