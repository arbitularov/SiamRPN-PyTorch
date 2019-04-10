import numpy as np

class Config(object):

    '''config for data.py'''
    template_img_size  = 127
    detection_img_size = 271
    score_size = int((detection_img_size - template_img_size) / 8 + 1)
    out_feature = 19
    max_inter   = 80
    fix_former_3_layers = True
    pretrained_model = '/Users/arbi/Desktop/alexnet.pth'

    total_stride = 8
    anchor_base_size = 8
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])

    valid_scope = int((detection_img_size - template_img_size) / total_stride / 2)
    anchor_valid_scope = 2 * valid_scope + 1
    pos_threshold = 0.6
    neg_threshold = 0.3

    context = 0.5
    penalty_k = 0.055
    window_influence = 0.42
    eps = 0.01

    '''config for net.py'''
    num_pos = 16
    num_neg = 48
    lamb    = 5

    ohem_pos = False
    ohem_neg = False
    ohem_reg = False

    '''config for train_siamrpn.py'''
    epoches = 200
    train_epoch_size = 1000
    val_epoch_size = 1000
    lr = 1e-6

    weight_decay = 0.0005
    momentum = 0.9

    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num    = len(anchor_scales) * len(anchor_ratios)


config = Config()
