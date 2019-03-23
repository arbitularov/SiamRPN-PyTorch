# -*- coding: utf-8 -*-

# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59

# @Last Modified by:   Arbi Tularov
# @Last Modified time: 2019-02-23 12:56:23

import os
import os.path as osp
import random
import time
import sys; sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from PIL import Image, ImageOps, ImageStat, ImageDraw
from data_loader import TrainDataLoader
from net import TrackerSiamRPN
from torch.nn import init
from shapely.geometry import Polygon
import json

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/Users/arbi/Desktop/val', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')
parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')


def main():

    """parameter initialization"""
    args = parser.parse_args()
    exp_name_dir = experiment_name_dir(args.experiment_name)

    """Load the parameters from json file"""
    json_path = os.path.join(exp_name_dir, 'parameters.json')
    assert os.path.isfile(json_path), ("No json configuration file found at {}".format(json_path))
    with open(json_path) as data_file:
        params = json.load(data_file)

    """ train dataloader """
    data_loader = TrainDataLoader(args.train_path, params)

    """ compute max_batches """
    for root, dirs, files in os.walk(args.train_path):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            args.max_batches += len(os.listdir(dir_path))

    """ Model on gpu """
    model = TrackerSiamRPN(params)
    #model = model.cuda()
    cudnn.benchmark = True

    """ load weights """
    init_weights(model)
    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        try:
            checkpoint = torch.load(args.checkpoint_path)
            start = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            start = 0
            init_weights(model)
    else:
        start = 0

    """ train phase """
    closses, rlosses, tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    steps = 0
    for epoch in range(start, params['epoches']):
        #cur_lr = adjust_learning_rate(params["lr"], optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__())
        for example in tqdm(range(10000)): # args.max_batches
            a = random.choice(index_list)
            #print('a', a)

            ret = data_loader.__get__(a)

            closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index, cur_lr = model.step(ret, epoch, backward=True)

            closs_ = closs.cpu().item()

            if np.isnan(closs_):
               sys.exit(0)

            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())
            tlosses.update(loss.cpu().item())
            steps+=1

            if example % 1000 == 0:
                print("Epoch:{:04d}\texample:{:06d}/{:06d}({:.2f})%\tlr:{:.7f}\tcloss:{:.4f}\trloss:{:.4f}\ttloss:{:.4f}".format((epoch+1), steps, args.max_batches, 100*(steps)/args.max_batches, cur_lr, closses.avg, rlosses.avg, tlosses.avg ))


        """save model"""
        model_save_dir_pth = '{}/model'.format(exp_name_dir)
        if not os.path.exists(model_save_dir_pth):
                os.makedirs(model_save_dir_pth)
        net_path = os.path.join(model_save_dir_pth, 'model_e%d.pth' % (epoch + 1))
        torch.save(model.net.state_dict(), net_path)

def experiment_name_dir(experiment_name):
    experiment_name_dir = 'experiments/{}'.format(experiment_name)
    if experiment_name == 'default':
        print('You are using "default" experiment, my advice to you is: Copy "default" change folder name and change settings in file "parameters.json"')
    else:
        print('You are using "{}" experiment'.format(experiment_name))
    return experiment_name_dir

def init_weights(model, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s' % init_type)
    model.net.apply(init_func)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

if __name__ == '__main__':
    main()
