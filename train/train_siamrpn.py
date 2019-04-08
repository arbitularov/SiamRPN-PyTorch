# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import init
from util import Util as util
from util import AverageMeter
from net import TrackerSiamRPN
from data import TrainDataLoader
from config import Config as config
from torch.utils.data import DataLoader
from got10k.datasets import ImageNetVID, GOT10k

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/home/arbi/desktop/GOT-10k', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')
# /home/arbi/desktop/GOT-10k # /Users/arbi/Desktop
# 'experiments/default/model/model_e74.pth'
def main():

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = AverageMeter.experiment_name_dir(args.experiment_name)

    '''model on gpu'''
    model = TrackerSiamRPN()

    '''setup train data loader'''
    name = 'All'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset = GOT10k(root_dir, subset='train')
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC2017_VID/ILSVRC'
        seq_dataset = ImageNetVID(root_dir, subset=('train'))
    elif name == 'All':
        root_dir_vid = '/home/arbi/desktop/ILSVRC2017_VID/ILSVRC'
        seq_datasetVID = ImageNetVID(root_dir_vid, subset=('train'))
        root_dir_got = args.train_path
        seq_datasetGOT = GOT10k(root_dir_got, subset='train')
        seq_dataset = util.data_split(seq_datasetVID, seq_datasetGOT)
    print('seq_dataset', len(seq_dataset))

    train_data  = TrainDataLoader(seq_dataset, name)
    train_loader = DataLoader(  dataset    = train_data,
                                batch_size = 1,
                                shuffle    = True,
                                num_workers= 16,
                                pin_memory = True)

    '''setup val data loader'''
    name = 'All'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset_val = GOT10k(root_dir, subset='val')
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC2017_VID/ILSVRC'
        seq_dataset_val = ImageNetVID(root_dir, subset=('val'))
    elif name == 'All':
        root_dir_vid = '/home/arbi/desktop/ILSVRC2017_VID/ILSVRC'
        seq_datasetVID = ImageNetVID(root_dir_vid, subset=('val'))
        root_dir_got = args.train_path
        seq_datasetGOT = GOT10k(root_dir_got, subset='val')
        seq_dataset_val = util.data_split(seq_datasetVID, seq_datasetGOT)
    print('seq_dataset_val', len(seq_dataset_val))

    val_data  = TrainDataLoader(seq_dataset_val, name)
    val_loader = DataLoader(  dataset    = val_data,
                                batch_size = 1,
                                shuffle    = True,
                                num_workers= 16,
                                pin_memory = True)

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
        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        with tqdm(total=config.train_epoch_size) as progbar:
            for i, dataset in enumerate(train_loader):

                closs, rloss, loss, cur_lr = model.step(epoch, dataset, backward=True)

                closs_ = closs.cpu().item()

                if np.isnan(closs_):
                   sys.exit(0)

                closses.update(closs.cpu().item())
                rlosses.update(rloss.cpu().item())
                tlosses.update(loss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(closses.avg), rloss='{:05.3f}'.format(rlosses.avg), tloss='{:05.3f}'.format(tlosses.avg))

                progbar.update()

                if i >= config.train_epoch_size - 1:
                    '''save plot'''
                    closses.closs_array.append(closses.avg)
                    rlosses.rloss_array.append(rlosses.avg)
                    tlosses.loss_array.append(tlosses.avg)
                    '''steps.update(steps.count)
                    steps.steps_array.append(steps.count)

                    steps.plot(exp_name_dir)'''

                    '''save model'''
                    model.save(model, exp_name_dir, epoch)

                    break

        '''val phase'''
        with tqdm(total=config.val_epoch_size) as progbar:
            print('Val epoch {}/{}'.format(epoch+1, config.epoches))
            for i, dataset in enumerate(val_loader):

                closs, rloss, loss, cur_lr = model.step(epoch, dataset, backward=False)

                closs_ = closs.cpu().item()

                if np.isnan(closs_):
                   sys.exit(0)

                closses.update(closs.cpu().item())
                rlosses.update(rloss.cpu().item())
                tlosses.update(loss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(closses.avg), rloss='{:05.3f}'.format(rlosses.avg), tloss='{:05.3f}'.format(tlosses.avg))

                progbar.update()

                if i >= config.train_epoch_size - 1:
                    '''save plot'''
                    closses.closs_array_val.append(closses.avg)
                    rlosses.rloss_array_val.append(rlosses.avg)
                    tlosses.loss_array_val.append(tlosses.avg)
                    steps.update(steps.count)
                    steps.steps_array.append(steps.count)

                    steps.plot(exp_name_dir)

                    '''save model'''
                    '''model.save(model, exp_name_dir, epoch)'''

                    break

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
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # good for relu
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
