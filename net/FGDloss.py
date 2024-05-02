import torch
import torch.nn as nn
import torch.nn.functional as F

class GCBlock(nn.Module):
    def __init__(self, teacher_channels=256):
        super(GCBlock, self).__init__()

        self.conv_mask_s = nn.Conv3d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv3d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv3d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv3d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv3d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv3d(teacher_channels//2, teacher_channels, kernel_size=1))

        # self.reset_parameters()

    def forward(self, preds_S, preds_T):
        return self.get_rela_loss(preds_S, preds_T)
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height, depth = x.size()
        input_x = x
        # [N, C, H * W * D]
        input_x = input_x.view(batch, channel, height * width * depth)
        # [N, 1, C, H * W * D]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W, D]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W * D]
        context_mask = context_mask.view(batch, 1, height * width * depth)
        # [N, 1, H * W * D]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W * D, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context
    
    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss
    
    def reset_parameters(self):
        # kaiming_init(self.conv_mask_s, mode='fan_in')
        # kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

def FGD_loss(t_feature, s_feature, annotation):
    N, C, W, H, D = t_feature.shape
    # get_attention return Spatial attention and channel attention
    t_S, t_C = get_attention(t_feature, 0.5)
    s_S, s_C = get_attention(s_feature, 0.5)

    # attention loss
    attention_loss = torch.sum(torch.abs((s_C-t_C)))/C + torch.sum(torch.abs((s_S-t_S)))/(H*W*D)

    # feature loss
    Mask_fg = torch.zeros_like(t_S)
    Mask_bg = torch.ones_like(t_S)
    Mask_fg, Mask_bg = get_feature_loss_mask(Mask_fg, Mask_bg, annotation, t_feature.shape)

    fg_loss, bg_loss = get_feature_loss(s_feature, t_feature, Mask_fg, Mask_bg, t_C, t_S)

    # global loss: Is this necessary?
    # global_loss = GCBlockModel(t_feature, s_feature)

    return attention_loss, fg_loss, bg_loss#, global_loss

def get_attention(preds, temp):
    """ preds: Bs*C*W*H*D """
    N, C, W, H, D = preds.shape

    value = torch.abs(preds)
    # Bs*W*H*D
    fea_map = value.mean(axis=1, keepdim=True)
    S_attention = (W * H * D * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, W, H, D)

    # Bs*C
    channel_map = value.view(N, C, -1).mean(axis=2,keepdim=False)
    C_attention = C * F.softmax(channel_map/temp, dim=1)

    return S_attention, C_attention

def get_feature_loss_mask(Mask_fg, Mask_bg, gt_bboxes, img_metas):
    # feature map size = original image size/8
    gt_bboxes = torch.from_numpy(gt_bboxes)
    N, _, W, H, D = img_metas
    
    for i in range(N):
        new_bboxes = torch.ones_like(gt_bboxes[i])
        # gt -> [center x, y, z, w, h, d]
        # gt_bboxes_cord / feat_img_size * original_img_size
        # gt_bboxes_shape / feat_img_size * original_img_size / 2
        # feat_img_size = original_img_size / 8
        new_bboxes[:, 0] = gt_bboxes[i][:, 0]/8
        new_bboxes[:, 1] = gt_bboxes[i][:, 1]/8
        new_bboxes[:, 2] = gt_bboxes[i][:, 2]/8
        new_bboxes[:, 3] = gt_bboxes[i][:, 3]/16
        new_bboxes[:, 4] = gt_bboxes[i][:, 4]/16
        new_bboxes[:, 5] = gt_bboxes[i][:, 5]/16

        wmin = torch.floor(new_bboxes[:, 0]-new_bboxes[:, 3]).int()     # cx - w/2
        wmax = torch.ceil( new_bboxes[:, 0]+new_bboxes[:, 3]).int()     # cx + w/2
        hmin = torch.floor(new_bboxes[:, 1]-new_bboxes[:, 4]).int()     # cy - h/2
        hmax = torch.ceil( new_bboxes[:, 1]+new_bboxes[:, 4]).int()     # cy + h/2
        dmin = torch.floor(new_bboxes[:, 2]-new_bboxes[:, 5]).int()     # cz - d/2
        dmax = torch.ceil( new_bboxes[:, 2]+new_bboxes[:, 5]).int()     # cz + d/2

        area = 1.0/(wmax.view(1,-1)+1-wmin.view(1,-1))/(hmax.view(1,-1)+1-hmin.view(1,-1))/(dmax.view(1,-1)+1-dmin.view(1,-1))
        
        for j in range(len(gt_bboxes[i])):
            h_slice = slice(hmin[j], hmax[j]+1)
            w_slice = slice(wmin[j], wmax[j]+1)
            d_slice = slice(dmin[j], dmax[j]+1)
            Mask_fg[i][h_slice, w_slice, d_slice] = \
                    torch.maximum(Mask_fg[i][hmin[j]:hmax[j]+1, wmin[j]:wmax[j]+1,dmin[j]:dmax[j]+1], area[0][j])

        Mask_bg[i][Mask_fg[i] > 0] = 0
        sum_bg = torch.sum(Mask_bg[i])

        if sum_bg.item() != 0:
            Mask_bg[i] /= sum_bg


    return Mask_fg, Mask_bg

def get_feature_loss(preds_S, preds_T, Mask_fg, Mask_bg, t_C, t_S):
    loss_mse = nn.MSELoss(reduction='sum')

    Mask_fg = Mask_fg.unsqueeze(dim=1)
    Mask_bg = Mask_bg.unsqueeze(dim=1)

    # [4, 128] -> [4, 128, 1, 1, 1]
    t_C = t_C.unsqueeze(dim=-1)
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