class Config(object):

    template_img_size = 127
    detection_img_size = 271
    epoches = 200
    train_epoch_size = 9300
    stride = 8
    lr = 1e-5
    weight_decay = 0.0005
    momentum = 0.9
    context = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8]
    penalty_k = 0.055
    window_influence = 0.42
