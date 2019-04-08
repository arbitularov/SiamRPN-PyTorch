class Config(object):

    '''config for data.py'''
    template_img_size  = 127
    detection_img_size = 271
    out_feature = 19
    max_inter   = 80

    stride = 8
    context = 0.5
    ratios  = [0.33, 0.5, 1, 2, 3]
    scales  = [8]
    penalty_k = 0.055

    window_influence = 0.42
    eps = 0.01

    '''config for net.py'''

    '''config for train_siamrpn.py'''
    epoches = 200
    train_epoch_size = 50000
    val_epoch_size = 50000
    lr = 1e-5

    weight_decay = 0.0005
    momentum = 0.9
