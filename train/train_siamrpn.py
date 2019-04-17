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
from config import config
from net import TrackerSiamRPN
from data import TrainDataLoader
from torch.utils.data import DataLoader
from util import util, AverageMeter, SavePlot
from got10k.datasets import ImageNetVID, GOT10k

torch.manual_seed(1234) # config.seed


parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/Users/arbi/Desktop', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')
# /home/arbi/desktop/GOT-10k # /Users/arbi/Desktop
# 'experiments/default/model/model_e1.pth'
def main():

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = util.experiment_name_dir(args.experiment_name)

    '''model on gpu'''
    model = TrackerSiamRPN()
    model.net.init_weights()

    '''setup train data loader'''
    name = 'VID'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset = GOT10k(root_dir, subset='val')
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC2017_VID'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    elif name == 'All':
        root_dir_vid = '/home/arbi/desktop/ILSVRC2017_VID'
        seq_datasetVID = ImageNetVID(root_dir_vid, subset=('train'))
        root_dir_got = args.train_path
        seq_datasetGOT = GOT10k(root_dir_got, subset='train')
        seq_dataset = util.data_split(seq_datasetVID, seq_datasetGOT)
    print('seq_dataset', len(seq_dataset))

    train_data  = TrainDataLoader(seq_dataset, name)
    train_loader = DataLoader(  dataset    = train_data,
                                batch_size = 1,
                                shuffle    = True,
                                num_workers= 1,
                                pin_memory = True)

    '''setup val data loader'''
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset_val = GOT10k(root_dir, subset='val')
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC2017_VID'
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
                                shuffle    = False,
                                num_workers= 1,
                                pin_memory = True)

    '''load weights'''

    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model' in checkpoint.keys():
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu')['model'])
        else:
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
        #model.net.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))
        print('You are loading the model.load_state_dict')

    elif config.pretrained_model:
        #print("init with pretrained checkpoint %s" % config.pretrained_model + '\n')
        #print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.net.state_dict()
        model_dict.update(checkpoint)
        model.net.load_state_dict(model_dict)

    torch.cuda.empty_cache()

    '''train phase'''
    train_closses, train_rlosses, train_tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    val_closses, val_rlosses, val_tlosses = AverageMeter(), AverageMeter(), AverageMeter()

    train_val_plot = SavePlot(exp_name_dir, 'train_val_plot')

    for epoch in range(config.epoches):
        model.net.train()
        if config.fix_former_3_layers:
                util.freeze_layers(model.net)
        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        train_loss = []
        with tqdm(total=config.train_epoch_size) as progbar:
            for i, dataset in enumerate(train_loader):

                closs, rloss, loss = model.step(epoch, dataset,i,  train=True)

                closs_ = closs.cpu().item()

                if np.isnan(closs_):
                   sys.exit(0)

                train_closses.update(closs.cpu().item())
                train_rlosses.update(rloss.cpu().item())
                train_tlosses.update(loss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(train_closses.avg),
                                    rloss='{:05.3f}'.format(train_rlosses.avg),
                                    tloss='{:05.3f}'.format(train_tlosses.avg))

                progbar.update()
                train_loss.append(train_tlosses.avg)
                #print('i', i, 'config.train_epoch_size - 1', config.train_epoch_size - 1)

                if i >= config.train_epoch_size - 1:
                    '''save plot'''
                    #train_val_plot.update(train_tlosses.avg, train_label = 'total loss')

                    '''save model'''
                    model.save(model, exp_name_dir, epoch)

                    break

        train_loss = np.mean(train_loss)

        '''val phase'''
        val_loss = []
        with tqdm(total=config.val_epoch_size) as progbar:
            print('Val epoch {}/{}'.format(epoch+1, config.epoches))
            for i, dataset in enumerate(val_loader):

                val_closs, val_rloss, val_tloss = model.step(epoch, dataset, train=False)

                closs_ = val_closs.cpu().item()

                if np.isnan(closs_):
                    sys.exit(0)

                val_closses.update(val_closs.cpu().item())
                val_rlosses.update(val_rloss.cpu().item())
                val_tlosses.update(val_tloss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(val_closses.avg),
                                    rloss='{:05.3f}'.format(val_rlosses.avg),
                                    tloss='{:05.3f}'.format(val_tlosses.avg))

                progbar.update()

                val_loss.append(val_tlosses.avg)

                if i >= config.val_epoch_size - 1:
                    break

        val_loss = np.mean(val_loss)
        train_val_plot.update(train_loss, val_loss)
        print ('Train loss: {}, val loss: {}'.format(train_loss, val_loss))


if __name__ == '__main__':
    main()
