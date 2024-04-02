import sys

from net.layer import *
from single_config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
import math
from scipy.stats import norm
from net.blocks import *
from net.FGDloss import FGD_loss
affine = True

class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels, block=ECA_SC):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            PreBlock(1),
            PreBlock(24))

        self.forw1 = nn.Sequential(
            block(24, 48),
            block(48, 48))

        self.forw2 = nn.Sequential(
            block(48, 72),
            block(72, 72))

        self.forw3 = nn.Sequential(
            block(72, 96),
            block(96, 96))

        self.forw4 = nn.Sequential(
            block(96, 120),
            block(120, 120))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 注意back2、back3順序
            # 128 + 72 = 200
            block(200, 128),
            block(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            # 注意back2、back3順序
            #120+96=216
            block(216, 128),
            block(128, 128))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(120, 120, kernel_size=2, stride=2),
            nn.BatchNorm3d(120),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        # multi-scale aggregation
        self.MSA1 = MultiScalAggregation3fs([48,72,96])
        self.MSA2 = MultiScalAggregation3fs([72,96,120])
        self.MSA3 = MultiScalAggregation2fs([96,120])

        # self.conv = nn.Sequential(
        #     nn.Conv3d(128,64,kernel_size=3, padding='same'),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True))

    def forward(self, x):
        fs1 = self.preBlock(x)#24

        fs1_pool, _ = self.maxpool1(fs1)
        fs2 = self.forw1(fs1_pool)#48

        fs2_pool, _ = self.maxpool2(fs2)
        fs3 = self.forw2(fs2_pool)#72
        #out2 = self.drop(out2)

        fs3_pool, _ = self.maxpool3(fs3)
        fs4 = self.forw3(fs3_pool)#96

        fs4_pool, _ = self.maxpool4(fs4)
        fs5 = self.forw4(fs4_pool)#120
        #out4 = self.drop(out4)
        
        rev3 = self.MSA3([fs4,fs5])       # output = 6 * 6 * 6 * 120
        rev2 = self.MSA2([fs3,fs4,fs5])  # 12* 12* 12* 96
        rev1 = self.MSA1([fs2,fs3,fs4])  # 24* 24* 24* 72


        up3 = self.path1(rev3)             # 12*12*12*120
        comb3 = self.back3(torch.cat((up3, rev2), 1))#120+96 ->128
        up2 = self.path2(comb3)            # 24*24*24*128
        comb2 = self.back2(torch.cat((up2, rev1), 1))#128+72->128
        return [x, fs2, comb2], fs3

class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        
        return logits, deltas

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas

class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size'] 

    def forward(self, f, inputs, proposals):
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = int(p[0])
            center = p[2:5]
            side_length = p[5:8]
            c0 = center - side_length / 2 # left bottom corner
            c1 = c0 + side_length # right upper corner
            c0 = (c0 / self.scale).floor().long()
            c1 = (c1 / self.scale).ceil().long()
            minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            maximum = torch.LongTensor(
                np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

            c0 = torch.cat((c0.unsqueeze(0), minimum), 0)
            c1 = torch.cat((c1.unsqueeze(0), maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            # Slice 0 dim, should never happen
            if np.any((c1 - c0).cpu().data.numpy() < 1):
                print(p)
                print('c0:', c0, ', c1:', c1)
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class MsaNet(nn.Module):
    def __init__(self, cfg, mode='train',hes='OHEM'):
        super(MsaNet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.hes = hes
        self.feature_net = FeatureNet(config,1  ,128)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        if self.cfg['FGD'] and mode != 'FGD':
            self.teacher_net = MsaNet(cfg, mode='FGD')
            self.teacher_net.load_state_dict(torch.load(self.cfg['teacher'])['state_dict'])
            self.teacher_net.eval()
        self.init_weight()
        

    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        # features, feat_4 = self.feature_net(inputs)
        if self.mode in ['train', 'valid']:
            if self.cfg['FGD'] and self.mode == 'train':

                with torch.no_grad():
                    t_features, t_feat_4 = data_parallel(self.teacher_net.feature_net, (inputs))
                    # self.teacher_net(input, truth_boxes, truth_labels)
                    # t_features = self.teacher_net.fs
                    t_fs = t_features[-1]

                self.attention_loss, self.fg_loss, self.bg_loss = FGD_loss(t_fs, self.fs, truth_boxes)
            features, feat_4 = data_parallel(self.feature_net, (inputs))
            fs = features[-1]
            fs_shape = fs.shape
            # self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)
            self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

            b,D,H,W,_,num_class = self.rpn_logits_flat.shape

            self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
            self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)
            self.rpn_window = make_rpn_windows(fs_shape, self.cfg)

        elif self.mode == 'eval':
            input_shape = inputs.shape
            B,C,D,H,W = input_shape
            self.rpn_logits_flat = torch.tensor([]).cuda()
            self.rpn_deltas_flat = torch.tensor([]).cuda()
            self.rpn_window = np.empty((0,6))
            
            last_crop = 0
            # B, C, D, H, W = 1, 128, 32, 128, 128
            rpn_windows = make_rpn_windows([1, 128, 32, 128, 128], self.cfg)

            for i in range(math.ceil(D/64)-1):

                if i*64+128 >= D:
                    crop_input = inputs[:,:,-128:]
                    overlap_slice = (D-(i*64))//4
                    last_crop = 1
                    start_slice = D-128
                else:
                    crop_input = inputs[:,:,i*64: i*64+128]
                    overlap_slice = 8
                    start_slice = i*64

                with torch.no_grad():
                    features, _ = data_parallel(self.feature_net,(crop_input))
                    self.fs = features[-1]
                    fs_shape = self.fs.shape
                    crop_rpn_logits_flat, crop_rpn_deltas_flat = data_parallel(self.rpn, self.fs)
                
                b,d,_,_,_,_ = crop_rpn_logits_flat.shape

                crop_rpn_window = rpn_windows.copy() + [start_slice, 0, 0, 0, 0, 0]
                if i == 0:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,:24]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,:24]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[:24]
                elif last_crop == 1:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,overlap_slice//2:]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,overlap_slice//2:]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[overlap_slice//2:]
                else:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,8:24]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,8:24]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[8:24]
                crop_rpn_window = crop_rpn_window.reshape(-1, 6)

                crop_rpn_logits_flat = crop_rpn_logits_flat.view(b, -1, 1)
                crop_rpn_deltas_flat = crop_rpn_deltas_flat.view(b, -1, 6)

                self.rpn_logits_flat = torch.cat((self.rpn_logits_flat, crop_rpn_logits_flat), dim=1)
                self.rpn_deltas_flat = torch.cat((self.rpn_deltas_flat, crop_rpn_deltas_flat), dim=1)
                self.rpn_window = np.concatenate((self.rpn_window, crop_rpn_window), axis=0)


        self.rpn_proposals = []

        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)

        if self.mode in ['train', 'valid']:
            # self.rpn_proposals = torch.zeros((0, 8)).cuda()
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels)

            if self.use_rcnn:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels)

        #rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        self.mask_probs = []
        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                proposal = self.rpn_proposals[:, [0, 2, 3, 4, 5, 6, 7]].cpu().numpy().copy()
                proposal[:, 1:] = center_box_to_coord_box(proposal[:, 1:])
                proposal = proposal.astype(np.int64)
                proposal[:, 1:] = ext2factor(proposal[:, 1:], 4)
                proposal[:, 1:] = clip_boxes(proposal[:, 1:], inputs.shape[2:])
                # rcnn_crops = self.rcnn_crop(features, inputs, torch.from_numpy(proposal).cuda())
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                rcnn_crops = data_parallel(self.rcnn_crop, (features, inputs, torch.from_numpy(proposal).cuda()))
                # rcnn_crops = self.rcnn_crop(fs, inputs, self.rpn_proposals)
                # self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals, 
                                                                        self.rcnn_logits, self.rcnn_deltas)

                if self.mode in ['eval']:
                    #     Ensemble
                    fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals, self.rcnn_logits,
                                              self.rcnn_deltas)
                    if self.ensemble_proposals.shape[0] == fpr_res.shape[0]:
                        self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2

    def loss(self, targets=None):
        cfg  = self.cfg
    
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rcnn_stats = None
    
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode, hes=self.hes)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)
    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss

        if self.cfg['FGD'] and self.mode == 'train':
            self.total_loss = self.total_loss \
                            + self.attention_loss * 3e-5\
                            + self.fg_loss * 3e-5\
                            + self.bg_loss * 1.5e-5

            return self.total_loss, rpn_stats, rcnn_stats, [self.attention_loss, self.fg_loss, self.bg_loss]
    
        return self.total_loss, rpn_stats, rcnn_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def init_weight(self):
        for param in self.feature_net.parameters():
            if isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(param.weight)
            elif isinstance(param, nn.BatchNorm3d):
                nn.init.constant_(param.weight, 1)
                nn.init.constant_(param.bias, 0)    
        for param in self.rpn.parameters():
            if isinstance(param, nn.Conv3d):
                nn.init.kaiming_normal_(param.weight)
                nn.init.constant_(param.bias, 0)
        for param in self.rcnn_head.parameters():
            if isinstance(param, nn.Linear):
                nn.init.kaiming_normal_(param.weight)
                nn.init.constant_(param.bias, 0)

    def set_anchor_params(self, anchor_ids, anchor_params):
        self.anchor_ids = anchor_ids
        self.anchor_params = anchor_params

    def crf(self, detections):
        """
        detections: numpy array of detection results [b, z, y, x, d, h, w, p]
        """
        res = []
        config = self.cfg
        anchor_ids = self.anchor_ids
        anchor_params = self.anchor_params
        anchor_centers = []

        for a in anchor_ids:
            # category starts from 1 with 0 denoting background
            # id starts from 0
            cat = a + 1
            dets = detections[detections[:, -1] == cat]
            if len(dets):
                b, p, z, y, x, d, h, w, _ = dets[0]
                anchor_centers.append([z, y, x])
                res.append(dets[0])
            else:
                # Does not have anchor box
                return detections
        
        pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
        for cat in pred_cats:
            if cat - 1 not in anchor_ids:
                cat = int(cat)
                preds = detections[detections[:, -1] == cat]
                score = np.zeros((len(preds),))
                roi_name = config['roi_names'][cat - 1]

                for k, params in enumerate(anchor_params):
                    param = params[roi_name]
                    for i, det in enumerate(preds):
                        b, p, z, y, x, d, h, w, _ = det
                        d = np.array([z, y, x]) - np.array(anchor_centers[k])
                        prob = norm.pdf(d, param[0], param[1])
                        prob = np.log(prob)
                        prob = np.sum(prob)
                        score[i] += prob

                res.append(preds[score == score.max()][0])
            
        res = np.array(res)
        return res

    def freeze_featurenet_rpn(self, grad=False):
        for param in self.rpn.parameters():
            param.requires_grad = False
        for param in self.feature_net.parameters():
            param.requires_grad = False
        for param in self.rcnn_crop.modules():
            if isinstance(param, nn.Conv3d):
                param.reset_parameters()
        for param in self.rcnn_head.modules():
            if isinstance(param, nn.Conv3d):
                param.reset_parameters()

if __name__ == '__main__':
    import torchsummary
    net = FeatureNet(config)

    # Create the input tensor with specific dimensions
    input_tensor = torch.randn(1, 128, 128, 128)

    # Move to GPU if needed
    if torch.cuda.is_available():
        net = net.to('cuda')
        input_tensor = input_tensor.to('cuda')

    torchsummary.summary(net, (1, 128, 128, 128), device='cuda')
    print("summary")