import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast as autocast
import numpy as np
from net.layer.util import box_transform_inv_torch
from utils.util import DIoU


def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    return loss.sum()

def binary_cross_entropy_with_hard_negative_mining(logits, labels, weights, fg_threshold=0.5, num_hard=2):
    '''Return 
        cls_loss : classification loss with binary cross entropy, 
        pos_correct: TP, 
        pos_total: TP+FN (same to recall), 
        neg_correct: TN, 
        neg_total: FP+TN (this is not precision)
    '''
    # with autocast(enabled=False):
    # classify_loss = nn.BCELoss()
    classify_loss = nn.BCEWithLogitsLoss(reduction='sum')
    probs = torch.sigmoid(logits.detach())[:, 0].view(-1, 1)
    # all labeled positive (TP+FN)
    pos_idcs = (labels[:, 0] == 1) & (weights[:, 0] != 0)
    # 
    pos_prob = probs[pos_idcs, 0]
    pos_logits = logits[pos_idcs, 0]
    pos_labels = labels[pos_idcs, 0]

    # For those weights are zero, there are 2 cases,
    # 1. Because we first random sample num_neg negative boxes for OHEM
    # 2. Because those anchor boxes have some overlap with ground truth box,
    #    we want to maintain high sensitivity, so we do not count those as
    #    negative. It will not contribute to the loss
    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)
    neg_prob = probs[neg_idcs, 0]
    neg_logits = logits[neg_idcs, 0]
    neg_labels = labels[neg_idcs, 0]
    if num_hard > 0:
        neg_prob, neg_logits, neg_labels = OHEM(neg_prob, neg_logits, neg_labels, num_hard * len(pos_prob))

    tol_logits = torch.concatenate((pos_logits, neg_logits))
    tol_labels = torch.concatenate((pos_labels, neg_labels))
    
    pos_correct = torch.tensor(0.)
    pos_total = 0.
    if len(pos_prob) > 0:
        pos_correct = (pos_prob >= fg_threshold).sum()
        pos_total = len(pos_prob)

    # cls_loss = 0.5 * classify_loss(pos_logits, pos_labels.float()) + 0.5 * classify_loss(neg_logits, neg_labels.float())
    # cls loss like retinanet devided by num of pos anchors
    cls_loss = classify_loss(tol_logits, tol_labels.float())
    cls_loss = cls_loss / torch.clamp(pos_correct, min=1.0)
    neg_correct = (neg_prob < fg_threshold).sum()
    neg_total = len(neg_prob)
    return cls_loss, pos_correct, pos_total, neg_correct, neg_total

def weighted_focal_loss_with_logits_OHEM(logits, labels, weights, fg_threshold=0.5, gamma=2., alpha=0.25, num_hard=3):
    '''Return 
        cls_loss : classification loss with focal loss, 
        pos_correct: TP, 
        pos_total: TP+FN (recall), 
        neg_correct: TN, 
        neg_total: FP+TN (not precision)
    '''
    # breakpoint()
    probs     = torch.sigmoid(logits)               # P   ,  y=(0,1)
    pos_idcs = (labels[:, 0] == 1) & (weights[:, 0] != 0)    # y=1
    pos_probs = probs[pos_idcs, 0]                           # P=Pt,  y=1
    pos_weight = torch.where(pos_probs < 0.6, torch.tensor(4), torch.tensor(1)).cuda()
    # For those weights are zero, there are 2 cases,
    # 1. Because we first random sample num_neg negative boxes for OHEM
    # 2. Because those anchor boxes have some overlap with ground truth box,
    #    we want to maintain high sensitivity, so we do not count those as
    #    negative. It will not contribute to the loss
    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)    # y=0
    neg_probs = probs[neg_idcs, 0]                           # P,  y=0
    if num_hard > 0:
        neg_probs = fOHEM(neg_probs, num_hard * max(len(pos_probs), 1)) # OHEM pick larger P(y=0)
        # neg_probs = fOHEM(neg_probs, num_hard * max(len(pos_probs),1))

    pos_logprobs = torch.log(pos_probs)                      # log(Pt), y=1
    neg_logprobs = torch.log(1 - neg_probs)                  # log(1-P)=log(Pt), y=0

    # pos_probs = pos_probs.detach()
    # neg_probs = neg_probs.detach()
    pos_weight = (alpha)* ((1-pos_probs.detach()) ** gamma)* pos_weight
    neg_weight = (1-alpha)* ((neg_probs.detach()) ** gamma)
    pos_loss =  pos_weight * pos_logprobs 
    neg_loss = neg_weight * neg_logprobs 

    pos_correct = (pos_probs > fg_threshold).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (neg_probs < fg_threshold).sum()
    neg_total = len(neg_probs)
    loss = -1*(pos_loss.sum() + neg_loss.sum())/ torch.clamp(pos_correct, min=1.0)
    # breakpoint()
    return loss, pos_correct, pos_total, neg_correct, neg_total

def weighted_focal_loss_with_logits(logits, labels, weights, gamma=2., alpha=0.25):
    probs     = torch.sigmoid(logits)                   # Pt

    pos_logprobs = torch.log(probs[labels == 1])        # log(Pt), y=1
    neg_logprobs = torch.log(1 - probs[labels == 0])    # log(Pt), y=0
    pos_probs = probs[labels == 1]                      # Pt     , y=1
    neg_probs = 1 - probs[labels == 0]                  # 1-Pt   , y=0
    pos_weights = weights[labels == 1]
    neg_weights = weights[labels == 0]

    pos_probs = pos_probs.detach()
    neg_probs = neg_probs.detach()

    pos_loss = - pos_logprobs*(alpha) * (1 - pos_probs) ** gamma
    neg_loss = - neg_logprobs*(1-alpha) * (1 - neg_probs) ** gamma
    loss = ((pos_loss * pos_weights).sum() + (neg_loss * neg_weights).sum()) / (weights.sum() + 1e-4)

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] < 0.5).sum()
    neg_total = (labels == 0).sum()

    return loss, pos_correct, pos_total, neg_correct, neg_total

def OHEM(neg_output, neg_logits, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_logits = torch.index_select(neg_logits, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_logits, neg_labels

def fOHEM(neg_output, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    return neg_output

def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train', delta_sigma=3.0, hes='OHEM'):
    batch_size, num_windows, num_classes = logits.size()
    labels = labels.long()

    # Calculate classification score
    batch_size = batch_size*num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    # Make sure OHEM is performed only in training mode
    if hes == 'focal':
        rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
            weighted_focal_loss_with_logits(logits, labels, label_weights)
    elif hes == 'fOHEM':
        if mode in ['train']:
            num_hard = cfg['num_hard']
        else:
            num_hard = 10000000
        rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
            weighted_focal_loss_with_logits_OHEM(logits, labels, label_weights, gamma=1, alpha=0.7,
                                                 fg_threshold=cfg['rpn_fg_threshold'], num_hard=num_hard)
    else:
        if mode in ['train']:
            num_hard = cfg['num_hard']
        else:
            num_hard = 10000000
        rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
            binary_cross_entropy_with_hard_negative_mining(logits, labels, label_weights, 
                                                           fg_threshold=cfg['rpn_fg_threshold'], num_hard=num_hard)

    # rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
    #    weighted_focal_loss_with_logits(logits, labels, label_weights)

    # Calculate regression
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)
    # breakpoint()
    index = (labels != 0).nonzero()[:,0]
    loss_weights = np.array(cfg['box_reg_loss_weight'])
    loss_weights = loss_weights/loss_weights.sum() * len(loss_weights)
    # breakpoint()
    if len(index):
        deltas  = deltas[index]
        targets = targets[index]

        rpn_reg_loss = 0
        reg_losses = []
        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i], beta=(1.0/9.0)) * loss_weights[i]
            rpn_reg_loss += l
            reg_losses.append(l.data.item())
        rpn_reg_loss = rpn_reg_loss/6

        # windows = torch.from_numpy(windows).cuda()[index%len(windows)]
        # GT   = box_transform_inv_torch(windows, targets,cfg['box_reg_weight'])
        # pred = box_transform_inv_torch(windows, deltas, cfg['box_reg_weight'])
        # for t, p in zip(GT, pred):
        #     rpn_reg_loss += (1-DIoU(t[:3], t[3:], p[:3], p[3:]))
    else:
        reg_losses = torch.tensor((0.,0.,0.,0.,0.,0.)).cuda()
        rpn_reg_loss = reg_losses.mean()

    return rpn_cls_loss, rpn_reg_loss, [pos_correct, pos_total, neg_correct, neg_total,
                                        reg_losses[0], reg_losses[1], reg_losses[2],
                                        reg_losses[3], reg_losses[4], reg_losses[5]]
