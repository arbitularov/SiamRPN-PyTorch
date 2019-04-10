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
from config import config
from torch.utils.data import DataLoader
from got10k.datasets import ImageNetVID, GOT10k

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/Users/arbi/Desktop', metavar='DIR',help='path to dataset')
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
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset = GOT10k(root_dir, subset='val')
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
                                batch_size = 8,
                                shuffle    = True,
                                num_workers= 16,
                                pin_memory = True)

    '''setup val data loader'''
    name = 'GOT-10k'
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

    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        model.net.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))
        print('You are loading the model.load_state_dict')

    elif config.pretrained_model:
        print("init with pretrained checkpoint %s" % config.pretrained_model + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.net.state_dict()
        model_dict.update(checkpoint)
        model.net.load_state_dict(model_dict)

    '''train phase'''
    closses, rlosses, tlosses, steps = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for epoch in range(config.epoches):
        model.net.train()
        if config.fix_former_3_layers:
            if torch.cuda.device_count() > 1:
                util.freeze_layers(model.net.module)
            else:
                util.freeze_layers(model.net)
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

                    break

if __name__ == '__main__':
    main()
