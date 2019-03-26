import torch
import numpy as np

class Anchor_ms(object):

    '''stable version for anchor generator'''

    def __init__(self, params):

        self.anchors = self.create_anchors(params)   # xywh
        self.eps     = 0.01

    def create_anchors(self, params):
        response_sz = (params['detection_img_size'] - params['template_img_size']) // params['stride'] + 1 #19

        anchor_num = len(params['ratios']) * len(params['scales'])
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = params['stride'] * params['stride']
        ind = 0
        for ratio in params['ratios']:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in params['scales']:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * params['stride']
        xs, ys = np.meshgrid(
            begin + params['stride'] * np.arange(response_sz),
            begin + params['stride'] * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)
        return anchors

    # float
    def diff_anchor_gt(self, gt):
        eps = self.eps
        anchors, gt = self.anchors.copy(), gt.copy()
        diff = np.zeros_like(anchors, dtype = np.float32)
        diff[:,0] = (gt[0] - anchors[:,0])/(anchors[:,2] + eps)
        diff[:,1] = (gt[1] - anchors[:,1])/(anchors[:,3] + eps)
        diff[:,2] = np.log((gt[2] + eps)/(anchors[:,2] + eps))
        diff[:,3] = np.log((gt[3] + eps)/(anchors[:,3] + eps))
        return diff

    # float
    def center_to_corner(self, box):
        box = box.copy()
        #print('box', box)
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[:,0]=box[:,0]-(box[:,2]-1)/2
        box_[:,1]=box[:,1]-(box[:,3]-1)/2
        box_[:,2]=box[:,0]+(box[:,2]-1)/2
        box_[:,3]=box[:,1]+(box[:,3]-1)/2
        box_ = box_.astype(np.float32)
        #print('box_', box_)

        return box_

    # float
    def corner_to_center(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[:,0]=box[:,0]+(box[:,2]-box[:,0])/2
        box_[:,1]=box[:,1]+(box[:,3]-box[:,1])/2
        box_[:,2]=(box[:,2]-box[:,0])
        box_[:,3]=(box[:,3]-box[:,1])
        box_ = box_.astype(np.float32)
        return box_

    def pos_neg_anchor(self, gt, pos_num=16, neg_num=48, threshold_pos=0.5, threshold_neg=0.1):
        gt = gt.copy()
        gt_corner = self.center_to_corner(np.array(gt, dtype = np.float32).reshape(1, 4))
        an_corner = self.center_to_corner(np.array(self.anchors, dtype = np.float32))
        iou_value = self.iou(an_corner, gt_corner).reshape(-1) #(1445)
        max_iou   = max(iou_value)
        pos, neg  = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # pos
        pos_cand = np.argsort(iou_value)[::-1][:30]
        pos_index = np.random.choice(pos_cand, pos_num, replace = False)

        if max_iou > threshold_pos:
            pos[pos_index] = 1
            #print('pos[pos_index]', pos[pos_index])

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]
        neg_ind = np.random.choice(neg_cand, neg_num, replace = False)
        neg[neg_ind] = 1

        return pos, neg

    def iou(self,box1,box2):
        box1, box2 = box1.copy(), box2.copy()
        N=box1.shape[0]
        K=box2.shape[0]
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))           # box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))           # box1=[N,K,4]
        x_max=np.max(np.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=np.min(np.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=np.max(np.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=np.min(np.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        tb[np.where(tb<0)]=0
        lr[np.where(lr<0)]=0
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square

class TrainDataLoader(object):
    def __init__(self, img_dir_path, params):

        self.anchor_generator = Anchor_ms(params)
        self.anchors = self.anchor_generator.create_anchors(params) #centor
        self.ret = {}
        self.params = params

    def _generate_pos_neg_diff(self, detection_box):
        gt_box_in_detection = detection_box
        pos, neg = self.anchor_generator.pos_neg_anchor(gt_box_in_detection)
        diff     = self.anchor_generator.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1,1)), diff.reshape((-1, 4))
        class_target = np.array([-100.] * self.anchors.shape[0], np.int32)
        #print('self.anchors.shape[0]', self.anchors.shape[0])

        # pos
        pos_index = np.where(pos == 1)[0]
        #pos_index = []
        #print('pos_index', pos_index)
        pos_num = len(pos_index)
        self.ret['pos_anchors'] = np.array(self.anchors[pos_index, :], dtype=np.int32) if not pos_num == 0 else None
        if pos_num > 0:
            class_target[pos_index] = 1

        # neg
        neg_index = np.where(neg == 1)[0]
        #print('neg_index', neg_index)

        neg_num = len(neg_index)
        class_target[neg_index] = 0

        # draw pos and neg anchor box

        class_logits = class_target.reshape(-1, 1)
        pos_neg_diff = np.hstack((class_logits, diff))
        self.ret['pos_neg_diff'] = pos_neg_diff
        return pos_neg_diff

    def _tranform(self):
        """To Tensor"""
        pos_neg_diff = self.ret['pos_neg_diff'].copy()
        #print('pos_neg_diff', pos_neg_diff.shape)

        self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)

    def __get__(self, detection_box):

        self._generate_pos_neg_diff(detection_box)
        self._tranform()
        return self.ret

    def __len__(self):
        return 0 #len(self.sub_class_dir)

if __name__ == '__main__':
    # we will do a test for dataloader
    loader = TrainDataLoader('/home/song/srpn/dataset/simple_vot13', check = True)
    #print(loader.__len__())
    index_list = range(loader.__len__())
    for i in range(1000):
        ret = loader.__get__(random.choice(index_list))
        label = ret['pos_neg_diff'][:, 0].reshape(-1)
        pos_index = list(np.where(label == 1)[0])
        pos_num = len(pos_index)
        print(pos_index)
        print(pos_num)
        if pos_num != 0 and pos_num != 16:
            print(pos_num)
            sys.exit(0)
        print(i)
