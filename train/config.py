import numpy as np

class Config(object):

    '''config for train_siamrpn.py'''
    template_img_size  = 127
    detection_img_size = 255
    epoches = 200
    train_epoch_size = 1000
    val_epoch_size = 100

    start_lr = 3e-6
    end_lr = 1e-7
    warm_lr = 1e-3
    warm_scale = warm_lr/start_lr
    epoch = 50
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    momentum = 0.9
    weight_decay = 0.0005

    clip = 10                           # grad clip

    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num    = len(anchor_scales) * len(anchor_ratios) # 5
    score_size = int((detection_img_size - template_img_size) / 8 + 1)
    size = anchor_num * score_size * score_size

    '''config for data.py'''

    out_feature = 19
    max_inter   = 80
    fix_former_3_layers = True
    pretrained_model = '/home/arbi/desktop/alexnet.pth' #'/home/arbi/Загрузки/alexnet.pth' # '/Users/arbi/Desktop/alexnet.pth'

    total_stride = 8
    anchor_total_stride = total_stride
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

    max_translate = 12
    scale_resize = 0.15
    gray_ratio = 0.25
    exem_stretch = False

    '''config for net.py'''
    num_pos = 16
    num_neg = 48
    lamb    = 5

    ohem_pos = False
    ohem_neg = False
    ohem_reg = False


config = Config()
