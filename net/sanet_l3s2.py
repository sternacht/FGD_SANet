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
from scipy.stats import norm
from net import resnet_l3s2, cgnl_ls
import math

bn_momentum = 0.1
affine = True

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, config):
        super(FeatureNet, self).__init__()
        self.resnet50 = resnet_l3s2.resnet50()

        self.back1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.reduce1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        self.reduce4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1, out1, out2, out3, out4 = self.resnet50(x)

        out4 = self.reduce1(out4)
        rev3 = self.path1(out4)
        out3 = self.reduce2(out3)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        out2 = self.reduce3(out2)
        comb2 = self.back2(torch.cat((rev2, out2), 1))

        return [x1, out1, comb2], out1

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
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        deltas = self.deltas(out)
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

class MaskHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, detections, features):
        img, f_2, f_4 = features  

        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)

        _, _, D, H, W = img.shape
        out = []

        for detection in detections:
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            up1 = f_4[b, :, z_start / 4:z_end / 4, y_start / 4:y_end / 4, x_start / 4:x_end / 4].unsqueeze(0)
            up2 = self.up2(up1)
            up2 = self.back2(torch.cat((up2, f_2[b, :, z_start / 2:z_end / 2, y_start / 2:y_end / 2, x_start / 2:x_end / 2].unsqueeze(0)), 1))
            up3 = self.up3(up2)
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)
            up3 = self.back3(torch.cat((up3, im), 1))

            logits = getattr(self, 'logits' + str(int(cat)))(up3)
            logits = logits.squeeze()
 
            mask = Variable(torch.zeros((D, H, W))).cuda()
            mask[z_start:z_end, y_start:y_end, x_start:x_end] = logits
            mask = mask.unsqueeze(0)
            out.append(mask)

        out = torch.cat(out, 0)

        return out


def crop_mask_regions(masks, crop_boxes):
    out = []
    for i in range(len(crop_boxes)):
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
        m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
        out.append(m)
    
    return out


def top1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res

def random1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res


class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size, in_channels = 128):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size']

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True),
            nn.Conv3d(64, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.back2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))

    def forward(self, f, inputs, proposals):
        img, out1, comb2 = f
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        img = img.squeeze(0)
        out1 = out1.squeeze(0)
        comb2 = comb2.squeeze(0)

        crops = []
        for p in proposals:
            b, z_start, y_start, x_start, z_end, y_end, x_end = p

            # Slice 0 dim, should never happen
            c0 = np.array(torch.Tensor([z_start, y_start, x_start]))
            c1 = np.array(torch.Tensor([z_end, y_end, x_end]))
            if np.any((c1 - c0) < 1): #np.any((c1 - c0).cpu().data.numpy() < 1):
                # c0=c0+1
                # c1=c1+1
                for i in range(3):
                    if c1[i] == 0:
                        c1[i] = c1[i] + 4
                    if c1[i] - c0[i] == 0:
                        c1[i] = c1[i] + 4
                print(p)
                print('c0:', c0, ', c1:', c1)
            z_end, y_end, x_end = c1

            fe1 = comb2[int(b), :, int(z_start / 4):int(z_end / 4), int(y_start / 4):int(y_end / 4), int(x_start / 4):int(x_end / 4)].unsqueeze(0)
            fe1_up = self.up2(fe1)

            fe2 = self.back2(torch.cat((fe1_up, out1[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)), 1))
            # fe2_up = self.up3(fe2)
            #
            # im = img[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)
            # up3 = self.back3(torch.cat((fe2_up, im), 1))
            # crop = up3.squeeze()

            crop = fe2.squeeze()
            # crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops
    
class SANet_L3S2(nn.Module):
    def __init__(self, cfg, mode='train',hes='OHEM'):
        super(SANet_L3S2, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.hes = hes
        self.feature_net = FeatureNet(config)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        if self.cfg['FGD'] and mode != 'FGD':
            self.teacher_net = SANet_L3S2(cfg, mode='FGD')
            self.teacher_net.load_state_dict(torch.load(self.cfg['teacher'])['state_dict'])
            self.teacher_net.eval()
        self.init_weight()
        
    def FGD_loss(self, t_feature, s_feature, annotation):
        # get_attention return Spatial attention and channel attention
        t_S, t_C = self.get_attention(t_feature, 0.5)
        s_S, s_C = self.get_attention(s_feature, 0.5)

        # attention loss
        attention_loss = torch.sum(torch.abs((s_C-t_C)))/len(s_C) + torch.sum(torch.abs((s_S-t_S)))/len(s_S)

        # feature loss
        Mask_fg = torch.zeros_like(t_S)
        Mask_bg = torch.ones_like(t_S)
        Mask_fg, Mask_bg = self.get_feature_loss_mask(Mask_fg, Mask_bg, annotation, t_feature.shape)

        fg_loss, bg_loss = self.get_feature_loss(s_feature, t_feature, Mask_fg, Mask_bg, t_C, t_S)

        # global loss: Is this necessary?
        # global_loss = self.GCBlockModel(t_feature, s_feature)

        return attention_loss, fg_loss, bg_loss#, global_loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W, D = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * D * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W, D)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention

    def get_feature_loss_mask(self, Mask_fg, Mask_bg, gt_bboxes, img_metas):
        # feature map size = original image size/8
        gt_bboxes = torch.from_numpy(gt_bboxes)
        N, _, H, W, D = img_metas
        img_W = W*8
        img_H = H*8
        img_D = D*8
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_W*W
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_W*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_H*H
            new_boxxes[:, 4] = gt_bboxes[i][:, 4]/img_H*H
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_D*D
            new_boxxes[:, 5] = gt_bboxes[i][:, 5]/img_D*D

            wmin = torch.floor(new_boxxes[:, 0]).int()
            wmax = torch.ceil(new_boxxes[:, 3]).int()
            hmin = torch.floor(new_boxxes[:, 1]).int()
            hmax = torch.ceil(new_boxxes[:, 4]).int()
            dmin = torch.floor(new_boxxes[:, 2]).int()
            dmax = torch.ceil(new_boxxes[:, 5]).int()

            area = 1.0/(hmax.view(1,-1)+1-hmin.view(1,-1))/(wmax.view(1,-1)+1-wmin.view(1,-1))/(dmax.view(1,-1)+1-dmin.view(1,-1))
            for j in range(len(gt_bboxes[i])):
                h_slice = slice(hmin[j], hmax[j]+1)
                w_slice = slice(wmin[j], wmax[j]+1)
                d_slice = slice(dmin[j], dmax[j]+1)
                
                Mask_fg[i][h_slice, w_slice, d_slice] = \
                        torch.maximum(Mask_fg[i][hmin[j]:hmax[j]+1, wmin[j]:wmax[j]+1, dmin[j]:dmax[j]+1], area[0][j])

            Mask_bg[i][Mask_fg[i] > 0] = 0
            sum_bg = torch.sum(Mask_bg[i])

            if sum_bg.item() != 0:
                Mask_bg[i] /= sum_bg


        return Mask_fg, Mask_bg

    def get_feature_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, t_C, t_S):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        t_C = t_C.unsqueeze(dim=-1)
        t_C = t_C.unsqueeze(dim=-1)

        t_S = t_S.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(t_S))
        fea_t = torch.mul(fea_t, torch.sqrt(t_C))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(t_S))
        fea_s = torch.mul(fea_s, torch.sqrt(t_C))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss

    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):

        if self.mode in ['train', 'valid']:

            if self.cfg['FGD'] and self.mode == 'train':

                with torch.no_grad():
                    t_features, t_feat_4 = data_parallel(self.teacher_net.feature_net, (inputs))
                    # self.teacher_net(input, truth_boxes, truth_labels)
                    # t_features = self.teacher_net.fs
                    t_fs = t_features[-1]

                self.attention_loss, self.fg_loss, self.bg_loss = self.FGD_loss(t_fs, self.fs, truth_boxes)

            features, feat_4 = data_parallel(self.feature_net, (inputs))
            self.fs = features[-1]
            fs_shape = self.fs.shape
            # self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)
            self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, self.fs)

            b,D,H,W,_,num_class = self.rpn_logits_flat.shape

            self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
            self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)
            self.rpn_window = make_rpn_windows(fs_shape, self.cfg)

        elif self.mode == 'eval':
            input_shape = inputs.shape
            B,C,D,H,W = input_shape
            # breakpoint()
            self.rpn_logits_flat = torch.tensor([]).cuda()
            self.rpn_deltas_flat = torch.tensor([]).cuda()
            self.rpn_window = np.empty((0,6))
            
            last_crop = 0

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

    def init_weight(self):
        for param in self.feature_net.parameters():
            if isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(param.weight)
            elif isinstance(param, nn.BatchNorm3d):
                nn.init.constant_(param.weight, 1)
                nn.init.constant_(param.bias, 0)    
        for param in self.rpn.parameters():
            if isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(param.weight)
            elif isinstance(param, nn.BatchNorm3d):
                nn.init.constant_(param.weight, 1)
                nn.init.constant_(param.bias, 0)    
        for param in self.rcnn_head.parameters():
            if isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(param.weight)
            elif isinstance(param, nn.BatchNorm3d):
                nn.init.constant_(param.weight, 1)
                nn.init.constant_(param.bias, 0)

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

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

