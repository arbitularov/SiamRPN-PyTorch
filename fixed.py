import re
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Fixed GOT-10k Dataset')
parser.add_argument('--dataset_path', default='/Users/arbi/Desktop/val', metavar='DIR',help='path to dataset')
param = parser.parse_args()

sub_class_dir = [sub_class_dir for sub_class_dir in os.listdir(param.dataset_path) if os.path.isdir(os.path.join(param.dataset_path, sub_class_dir))]

array_error = []

for name_dir in tqdm(sub_class_dir):

    meta = open("{}/{}/meta_info.ini".format(param.dataset_path, name_dir), "r")

    read_meta = meta.readlines()

    w_and_h = re.findall(r'\d+', read_meta[10])
    meta.close()

    groundtruth = open("{}/{}/groundtruth.txt".format(param.dataset_path, name_dir), "r")
    read_groundtruth = groundtruth.readlines()
    count_gt = len(read_groundtruth)
    groundtruth.close()

    groundtruth_write = open("{}/{}/groundtruth.txt".format(param.dataset_path, name_dir), "w")

    groundtruth_array = []
    for i2, name_gt in enumerate(read_groundtruth):

        gt = [abs(int(float(i))) for i in name_gt.strip('\n').split(',')]
        w = gt[0]+gt[2]
        h = gt[1]+gt[3]

        if w > int(w_and_h[0]) and h > int(w_and_h[1]):
            print('i2', i2+1,'w and h')
            info = 'w and h {}, img: {}, img_size_h: {} < ymax = {} + {} = {} and img_size_w: {} < xmax = {} + {} = {} '.format(name_dir, i2+1, w_and_h[0], gt[0], gt[2], w, w_and_h[1], gt[1], gt[3], h )
            array_error.append(info)
            w_fixed = gt[2] - (w - int(w_and_h[0]))
            h_fixed = gt[3] - (h - int(w_and_h[1]))
            gt_fixed = '{}.0000,{}.0000,{}.0000,{}.0000'.format(gt[0], gt[1], w_fixed, h_fixed)
            groundtruth_array.append(gt_fixed)

        elif w > int(w_and_h[0]):
            #print('i2', i2+1,'just w')
            info = 'just w {}, img: {}, img_size: {} < xmax = {} + {} = {}'.format(name_dir, i2+1, w_and_h[0], gt[0], gt[2], w)
            array_error.append(info)
            w_fixed = gt[2] - (w - int(w_and_h[0]))
            gt_fixed = '{}.0000,{}.0000,{}.0000,{}.0000'.format(gt[0], gt[1], w_fixed, gt[3])
            groundtruth_array.append(gt_fixed)

        elif h > int(w_and_h[1]):
            #print('i2', i2+1,'just h')
            info = 'just w {}, img: {}, img_size: {} < ymax = {} + {} = {}'.format(name_dir, i2+1, w_and_h[1], gt[1], gt[3], h)
            array_error.append(info)
            h_fixed = gt[3] - (h - int(w_and_h[1]))
            gt_fixed = '{}.0000,{}.0000,{}.0000,{}.0000'.format(gt[0], gt[1], gt[2], h_fixed)
            groundtruth_array.append(gt_fixed)

        else:
            #print('i2', i2+1,'all it\'s ok')
            gt_fixed = '{}.0000,{}.0000,{}.0000,{}.0000'.format(gt[0], gt[1], gt[2], gt[3])
            groundtruth_array.append(gt_fixed)
    try:
        for l in groundtruth_array:
            groundtruth_write.write('{}\n'.format(l))
    finally:
        groundtruth_write.close()

new_file = open("new_file.txt", "w")
try:
    for i in array_error:
        new_file.write('{}\n'.format(i))
finally:
    new_file.close()
