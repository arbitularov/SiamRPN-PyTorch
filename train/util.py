import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class Util(object):

    def data_split(seq_datasetVID, seq_datasetGOT):
        seq_dataset = []
        for i in seq_datasetVID:
            seq_dataset.append(i)

        for i, data in enumerate(seq_datasetGOT):
            seq_dataset.append(data)
            if i >= 8600:
                break
        return seq_dataset

    def generate_anchors(self, total_stride, base_size, scales, ratios, score_size):
        print(total_stride, base_size, scales, ratios, score_size)
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

    # freeze layers
    def freeze_layers(model):
        print('------------------------------------------------------------------------------------------------')
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
        print("fixed layers:")
        print(model.featureExtract[:10])

util = Util()

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()
        self.plot_c = 0

    steps_array = []
    loss_array  = []
    closs_array = []
    rloss_array = []

    loss_array_val  = []
    closs_array_val = []
    rloss_array_val = []

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

    '''setup plot'''
    def plot(self,  exp_name_dir,
                    step   = steps_array,
                    train_loss   = loss_array,
                    train_closs  = closs_array,
                    train_rloss  = rloss_array,

                    val_loss   = loss_array_val,
                    val_closs  = closs_array_val,
                    val_rloss  = rloss_array_val,

                    title  = "Siamese RPN",
                    ylabel = 'error',
                    xlabel = 'epoch',
                    show   = False):
        plt.plot(step, train_loss, 'r', label='train loss', color='blue')
        plt.plot(step, train_closs, 'r', label='train closs', color='red')
        plt.plot(step, train_rloss, 'r', label='train rloss', color='black')

        plt.plot(step, val_loss, 'k--', label='val loss', color='blue')
        plt.plot(step, val_closs, 'k--', label='val closs', color='red')
        plt.plot(step, val_rloss, 'k--', label='val rloss', color='black')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if self.plot_c == 0:
            plt.legend()
            self.plot_c +=1

        '''save plot'''
        plt.savefig("{}/test.png".format(exp_name_dir))
        if show:
            plt.show()

    def experiment_name_dir(experiment_name):
        experiment_name_dir = 'experiments/{}'.format(experiment_name)
        if experiment_name == 'default':
            print('You are using "default" experiment, my advice to you is: Copy "default" change folder name and change settings in file "parameters.json"')
        else:
            print('You are using "{}" experiment'.format(experiment_name))
        return experiment_name_dir
