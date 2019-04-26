import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util import util

def rpn_cross_entropy_old(input, target):
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu])
    return loss


def rpn_cross_entropy_balance_old(input, target, num_pos, num_neg):
    cal_index_pos = np.array([], dtype=np.int64)
    cal_index_neg = np.array([], dtype=np.int64)
    for batch_id in range(target.shape[0]):
        print(target[batch_id])
        pos_index = np.random.choice(np.where(target[batch_id].cpu() == 1)[0], num_pos)
        neg_index = np.random.choice(np.where(target[batch_id].cpu() == 0)[0], num_neg)
        cal_index_pos = np.append(cal_index_pos, batch_id * target.shape[1] + pos_index)
        cal_index_neg = np.append(cal_index_neg, batch_id * target.shape[1] + neg_index)
    pos_loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index_pos], target=target.flatten()[cal_index_pos],
                               reduction='sum') / cal_index_pos.shape[0]
    neg_loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index_neg], target=target.flatten()[cal_index_neg],
                               reduction='sum') / cal_index_neg.shape[0]
    loss = (pos_loss + neg_loss) / 2
    # loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.flatten()[cal_index])
    return loss

def rpn_smoothL1_old(input, target, label):
    pos_index = np.where(label.cpu() == 1)
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index])
    return loss

def rpn_cross_entropy_balance(input, target, num_pos, num_neg, anchors, ohem_pos=None, ohem_neg=None):
    cuda = torch.cuda.is_available()
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))

        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()

        if ohem_pos:
            if len(pos_index) > 0:
                pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
                                               target=target[batch_id][pos_index], reduction='none')
                selected_pos_index = util.nms(anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
                pos_loss_bid_final = pos_loss_bid[selected_pos_index]
            else:
                if cuda:
                    pos_loss_bid = torch.FloatTensor([0]).cuda()
                else:
                    pos_loss_bid = torch.FloatTensor([0])
                pos_loss_bid_final = pos_loss_bid
        else:
            pos_index_random = random.sample(pos_index, min_pos)
            if len(pos_index) > 0:
                pos_loss_bid_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random], reduction='none')
            else:
                if cuda:
                    pos_loss_bid_final = torch.FloatTensor([0]).cuda()
                else:
                    pos_loss_bid_final = torch.FloatTensor([0])

        if ohem_neg:
            if len(pos_index) > 0:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduction='none')
                selected_neg_index = util.nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
            else:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduction='none')
                selected_neg_index = util.nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
        else:
            if len(pos_index) > 0:
                neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), min_neg)
                #neg_index_random = np.where(target[batch_id].cpu() == 0)[0].tolist()
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduction='none')
            else:
                neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), num_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduction='none')
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all.append(loss_bid)
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def rpn_smoothL1(input, target, label, num_pos=16, ohem=None):
    cuda = torch.cuda.is_available()
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)
        if ohem:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index], reduction='none')
                sort_index = torch.argsort(loss_bid.mean(1))
                loss_bid_ohem = loss_bid[sort_index[-num_pos:]]
            else:
                if cuda:
                    loss_bid_ohem = torch.FloatTensor([0]).cuda()[0]
                else:
                    loss_bid_ohem = torch.FloatTensor([0])[0]

            loss_all.append(loss_bid_ohem.mean())
        else:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index])
            else:
                if cuda:
                    loss_bid = torch.FloatTensor([0]).cuda()[0]
                else:
                    loss_bid = torch.FloatTensor([0])[0]

            loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss
