import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):

        cout, rout = predictions

        """ class """

        class_pred, class_target = cout, targets[:,0].long()
        pos_index , neg_index    = list(np.where(class_target.cpu() == 1)[0]), list(np.where(class_target.cpu() == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)

        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, reduction='none')

        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred = rout
        reg_target = targets[:, 1:]

        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none')

        rloss = torch.div(torch.sum(rloss, dim = 1), 4)

        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        loss = closs + 10*rloss

        return closs, rloss, loss #, reg_pred, reg_target, pos_index, neg_index
