import sys
import warnings
from net.layer import *
from net.MSANet import MsaNet
from single_config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes, crop_size
from torch.nn.parallel import data_parallel
import random
import math
import numpy as np
from scipy.stats import norm
from net.blocks import *
from net.FGDloss import FGD_loss
affine = True

class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels, block=ECA_SC):
        super(FeatureNet, self).__init__()

        self.forw1 = nn.Sequential(
            block(1, 24),
            block(24, 24))

        self.forw2 = nn.Sequential(
            block(24, 48),
            block(48, 48))

        self.forw3 = nn.Sequential(
            block(48, 72),
            block(72, 72))

        self.forw4 = nn.Sequential(
            block(72, 96),
            block(96, 96))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 96 + 72 = 168
            block(168, 128),
            block(128, 128))


        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(96, 96, kernel_size=2, stride=2),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True))

        # self.conv = nn.Sequential(
        #     nn.Conv3d(128,64,kernel_size=3, padding='same'),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True))

    def forward(self, x):
        fs1 = self.forw1(x)#128**3 *24

        fs1_pool, _ = self.maxpool1(fs1)#64**3
        fs2 = self.forw2(fs1_pool)#48

        fs2_pool, _ = self.maxpool2(fs2)#32**3
        fs3 = self.forw3(fs2_pool)#72
        #out2 = self.drop(out2)

        fs3_pool, _ = self.maxpool3(fs3)#16**3
        fs4 = self.forw4(fs3_pool)#96
        #out4 = self.drop(out4)
        
        rev2 = fs4  #16**3 *96
        rev1 = fs3  #32**3 *72

        up2 = self.path2(rev2)            # 16**3 *96
        comb2 = self.back2(torch.cat((up2, rev1), 1))   # 96+72->128
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

class SANet_ECASC_R(nn.Module):
    def __init__(self, cfg, mode='train',hes='OHEM'):
        super(SANet_ECASC_R, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.hes = hes
        self.feature_net = FeatureNet(config, 1 ,128)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        if self.cfg['FGD'] and mode != 'FGD':
            print(f"load teacher model weight {self.cfg['teacher']}")
            self.teacher_net = MsaNet(cfg, mode='FGD')
            self.teacher_net.load_state_dict(torch.load(self.cfg['teacher'])['state_dict'])
            self.teacher_net.eval()
            self.loss_weight = cfg['FGD_loss_weight']
        # self.feature_net.apply(self.init_weight)
        # self.rpn.apply(self.init_weight)
        # self.rcnn_head.apply(self.init_weight)
        
    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None, lobe_info=None):
        # features, feat_4 = self.feature_net(inputs)
        if self.mode in ['train', 'valid']:
            features, feat_4 = data_parallel(self.feature_net, (inputs))
            fs = features[-1]
            fs_shape = fs.shape
            self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

            b,D,H,W,_,num_class = self.rpn_logits_flat.shape

            self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
            self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)
            self.rpn_window = make_rpn_windows(fs_shape, self.cfg)

            if self.cfg['FGD'] and self.mode == 'train':

                with torch.no_grad():
                    t_features, t_feat_4 = data_parallel(self.teacher_net.feature_net, (inputs))
                    t_fs = t_features[-1]

                self.attention_loss, self.fg_loss, self.bg_loss = FGD_loss(t_fs, fs, truth_boxes)

        elif self.mode == 'eval':
            # breakpoint()
            if lobe_info is not None:
                # crop the input with lobe and test whole cropped image
                Ds, De, Hs, He, Ws, We = lobe_info[0]
                crop_inputs = inputs[:,:, Ds:De, Hs:He, Ws:We]
                # crop_inputs = inputs
                B,C,D,H,W = crop_inputs.shape
                # breakpoint()
                self.rpn_windows = make_rpn_windows([1, 128, D//4, H//4, W//4], self.cfg)
                self.rpn_window = self.rpn_windows + [Ds, Hs, Ws, 0, 0, 0]
                with torch.no_grad():
                    features, _ = data_parallel(self.feature_net,(crop_inputs))
                    fs = features[-1]
                    fs_shape = fs.shape
                    self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)
                b,d,h,w,a,_ = self.rpn_logits_flat.shape
                self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
                self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)

            else:
                # turn input image into small block and test each of them
                B,C,D,H,W = inputs.shape
                self.rpn_logits_flat = torch.tensor([]).cuda()
                self.rpn_deltas_flat = torch.tensor([]).cuda()
                self.rpn_window = np.empty((0,6))
                last_crop = 0
                # B, C, D, H, W = 1, 128, 32, 128, 128
                rpn_windows = make_rpn_windows([1, 128, 32, 128, 128], self.cfg)

                for i in range(math.ceil(D/64)-1):
                    hw_block = 1 
                    block_size = 512
                    hw_stride = 512-(block_size*hw_block - 512)/(hw_block-1) if hw_block > 1 else 0
                    if not isinstance(hw_stride, int):
                        warnings.warn(f'hw_stride should be int')
                        hw_stride = int(hw_stride)
                    if hw_stride not in crop_size.keys():
                        raise ValueError(f'hw_block:{hw_block} and block_size:{block_size} not set')
                    for hw in range(hw_block**2):
                        h_idx = hw%hw_block
                        w_idx = hw//hw_block
                        crop_h = crop_size[hw_stride][h_idx]
                        crop_w = crop_size[hw_stride][w_idx]
                        # breakpoint()
                        if i*64+128 >= D:
                            crop_input = inputs[:,:,-128:, h_idx*hw_stride:h_idx*hw_stride+block_size, w_idx*hw_stride:w_idx*hw_stride+block_size]
                            overlap_slice = (D-(i*64))//4
                            last_crop = 1
                            start_slice = D-128
                        else:
                            crop_input = inputs[:,:,i*64: i*64+128, h_idx*hw_stride:h_idx*hw_stride+block_size, w_idx*hw_stride:w_idx*hw_stride+block_size]
                            overlap_slice = 16
                            start_slice = i*64
                        with torch.no_grad():
                            features, _ = data_parallel(self.feature_net,(crop_input))
                            fs = features[-1]
                            fs_shape = fs.shape
                            crop_rpn_logits_flat, crop_rpn_deltas_flat = data_parallel(self.rpn, fs)

                        b,d,h,w,a,_ = crop_rpn_logits_flat.shape

                        crop_rpn_window = rpn_windows.copy() + [start_slice, h_idx*hw_stride, w_idx*hw_stride, 0, 0, 0]

                        if i == 0:
                            crop_rpn_logits_flat = crop_rpn_logits_flat[:,:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_window = crop_rpn_window.reshape(d, h, w, -1, 6)[:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                        elif last_crop == 1:
                            crop_rpn_logits_flat = crop_rpn_logits_flat[:,overlap_slice//2:,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,overlap_slice//2:,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_window = crop_rpn_window.reshape(d, h, w, -1, 6)[overlap_slice//2:,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                        else:
                            crop_rpn_logits_flat = crop_rpn_logits_flat[:,8:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,8:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
                            crop_rpn_window = crop_rpn_window.reshape(d, h, w, -1, 6)[8:24,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]

                        crop_rpn_logits_flat = crop_rpn_logits_flat.contiguous().view(b, -1, 1)
                        crop_rpn_deltas_flat = crop_rpn_deltas_flat.contiguous().view(b, -1, 6)
                        crop_rpn_window = crop_rpn_window.reshape(-1, 6)

                        self.rpn_logits_flat = torch.cat((self.rpn_logits_flat, crop_rpn_logits_flat), dim=1)
                        self.rpn_deltas_flat = torch.cat((self.rpn_deltas_flat, crop_rpn_deltas_flat), dim=1)
                        self.rpn_window = np.concatenate((self.rpn_window, crop_rpn_window), axis=0)


        self.rpn_proposals = []
        # breakpoint()
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
        # breakpoint()
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode, hes=self.hes)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)
    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss

        if self.cfg['FGD'] and self.mode == 'train':
            self.attention_loss = self.attention_loss * self.loss_weight[0]
            self.fg_loss = self.fg_loss * self.loss_weight[1]
            self.bg_loss = self.bg_loss * self.loss_weight[2]
            self.total_loss = self.total_loss \
                            + self.attention_loss\
                            + self.fg_loss\
                            + self.bg_loss

            return self.total_loss, rpn_stats, rcnn_stats, [self.attention_loss, self.fg_loss, self.bg_loss]
    
        return self.total_loss, rpn_stats, rcnn_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def init_weight(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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