import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class Util(object):

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
        # (5,4x225) to (225x5,4)

        ori = - (score_size // 2) * total_stride # 8 * 8 = 64
        #print('score_size', score_size)
        #print('total_stride', total_stride)
        #print('ori', ori)

        # the left displacement
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        #print('xx, yy ', xx, yy )
        # (15,15)
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        # (15,15) to (225,1) to (5,225) to (225x5,1)
        #print('xx1, yy1', xx, yy )

        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        #print('anchor', anchor)
        return anchor

    # freeze layers
    def freeze_layers(self, model):
        #print('------------------------------------------------------------------------------------------------')
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
        #print("fixed layers:")
        #print(model.featureExtract[:10])

    def experiment_name_dir(self, experiment_name):
        experiment_name_dir = 'experiments/{}'.format(experiment_name)
        if experiment_name == 'default':
            print('You are using "default" experiment, my advice to you is: Copy "default" change folder name and change settings in file "parameters.json"')
        else:
            print('You are using "{}" experiment'.format(experiment_name))
        return experiment_name_dir

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
                        val_label = val_label)

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
