# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import init
from util import AverageMeter
from net import TrackerSiamRPN
from data import TrainDataLoader
from parameters import Config as config
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/Users/arbi/Desktop/val', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')

#../weights-0690000.pth.tar #../model_e25.pth

def main():

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = AverageMeter.experiment_name_dir(args.experiment_name)

    '''model on gpu'''
    model = TrackerSiamRPN()

    '''setup data loader'''
    data_loader = TrainDataLoader(args.train_path)

    '''load weights'''
    init_weights(model)

    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        try:
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))
            print('You are loading the model.load_state_dict')
        except:
            init_weights(model)

    '''train phase'''
    closses, rlosses, tlosses, steps = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for epoch in range(config.epoches):
        for example in tqdm(range(10)):

            index_list = range(data_loader.__len__())
            closs, rloss, loss, cur_lr = model.step(epoch, data_loader, example, index_list, backward=True)

            closs_ = closs.cpu().item()

            if np.isnan(closs_):
               sys.exit(0)

            closses.update(closs.cpu().item())
            closses.closs_array.append(closses.avg)

            rlosses.update(rloss.cpu().item())
            rlosses.rloss_array.append(rlosses.avg)

            tlosses.update(loss.cpu().item())
            tlosses.loss_array.append(tlosses.avg)

            steps.update(tlosses.count)
            steps.steps_array.append(steps.count)

            if example % 1 == 0:
                print("Train epoch:{:04d}\texample:{:06d}/{:06d}({:.2f})%\tlr:{:.7f}\tcloss:{:.4f}\trloss:{:.4f}\ttloss:{:.4f}".format((epoch+1),
                        steps.count, data_loader._max_batches(), 100*(steps.count)/data_loader._max_batches(),
                        cur_lr, closses.avg, rlosses.avg, tlosses.avg ))

        '''save plot'''
        steps.plot(exp_name_dir)

        '''save model'''
        model.save(model, exp_name_dir, epoch)

def init_weights(model, init_type='normal', gain=0.02):
    def init_func(m):
        '''this will apply to each layer'''
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

if __name__ == '__main__':
    main()
