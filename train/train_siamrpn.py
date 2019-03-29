# -*- coding: utf-8 -*-
import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import init
from net import TrackerSiamRPN
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from data_loader import TrainDataLoader
from got10k.datasets import ImageNetVID, GOT10k

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/Users/arbi/Desktop/val', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')
parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

def main():

    '''setup dataset'''
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k']
    if name == 'GOT-10k':
        root_dir = 'data/GOT-10k'
        seq_dataset = GOT10k('/Users/arbi/Desktop/', subset='val')
    elif name == 'VID':
        root_dir = 'data/ILSVRC'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = experiment_name_dir(args.experiment_name)

    '''Load the parameters from json file'''
    json_path = os.path.join(exp_name_dir, 'parameters.json')
    assert os.path.isfile(json_path), ("No json configuration file found at {}".format(json_path))
    with open(json_path) as data_file:
        params = json.load(data_file)

    '''model on gpu'''
    model = TrackerSiamRPN(params)
    #model = model.cuda()
    cudnn.benchmark = True

    '''setup data loader'''
    data_loader = TrainDataLoader(model, seq_dataset, params)

    '''
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=8, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=4)
    '''

    '''compute max_batches'''
    for root, dirs, files in os.walk(args.train_path):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            args.max_batches += len(os.listdir(dir_path))

    '''load weights'''
    init_weights(model)

    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        try:
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage))
            print('You are loading the model.load_state_dict')
        except:
            init_weights(model)

    '''Data for plotting'''
    steps_array = []
    loss_array  = []
    closs_array = []
    rloss_array = []

    def plot(step, loss, closs, rloss, exp_name_dir, show=True):
        '''setup plot'''
        plt.plot(step, loss, 'r', label='loss', color='blue')
        plt.plot(step, closs, 'r', label='closs', color='red')
        plt.plot(step, rloss, 'r', label='rloss', color='black')
        plt.title("Siamese RPN")
        plt.ylabel('error')
        plt.legend()
        plt.xlabel('epoch')

        '''save plot'''
        plt.savefig("{}/test.png".format(exp_name_dir))
        if show:
            plt.show()

    '''train phase'''
    closses, rlosses, tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    start = 0
    steps = 0
    for epoch in range(start, params['epoches']):

        for example in tqdm(range(100)): # args.max_batches

            closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index, cur_lr = model.step(data_loader, epoch, backward=True)

            closs_ = closs.cpu().item()

            if np.isnan(closs_):
               sys.exit(0)

            closses.update(closs.cpu().item())
            rlosses.update(rloss.cpu().item())
            tlosses.update(loss.cpu().item())
            steps+=1

            if example % 1 == 0:
                print("Epoch:{:04d}\texample:{:06d}/{:06d}({:.2f})%\tlr:{:.7f}\tcloss:{:.4f}\trloss:{:.4f}\ttloss:{:.4f}".format((epoch+1), steps, args.max_batches, 100*(steps)/args.max_batches, cur_lr, closses.avg, rlosses.avg, tlosses.avg ))
                steps_array.append(steps)
                loss_array.append(tlosses.avg)
                closs_array.append(closses.avg)
                rloss_array.append(rlosses.avg)

        plot(steps_array, loss_array, closs_array, rloss_array, exp_name_dir)

        '''save model'''
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

if __name__ == '__main__':
    main()
