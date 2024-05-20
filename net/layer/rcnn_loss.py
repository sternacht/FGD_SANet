import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def rcnn_loss(logits, deltas, labels, targets, deltas_sigma=1.0):
    batch_size, num_class = logits.size(0),logits.size(1)

    # Weighted cross entropy for imbalance class distribution
    weight = torch.ones(num_class).cuda()
    total = len(labels)
    for i in range(num_class):
        num_pos = float((labels == i).sum())
        num_pos = max(num_pos, 1)
        weight[i] = total / num_pos

    weight = weight / weight.sum()
    rcnn_cls_loss = F.cross_entropy(logits, labels, weight=weight, size_average=True)

    # If multi-class classification, compute the confusion metric to understand the mistakes
    confusion_matrix = np.zeros((num_class, num_class))
    probs = F.softmax(logits, dim=1)
    v, cat = torch.max(probs, dim=1)
    for i in labels.nonzero():
        i = i.item()
        confusion_matrix[labels.long().detach()[i].item()][cat[i].detach().item()] += 1

    num_pos = len(labels.nonzero())

    if num_pos > 0:
        # one hot encode
        select = torch.zeros((batch_size,num_class)).cuda()
        select.scatter_(1, labels.view(-1,1), 1)
        select[:,0] = 0
        select = select.view(batch_size,num_class, 1).expand((batch_size, num_class, 6)).contiguous().byte()
        select = select.bool()
        deltas = deltas.view(batch_size, num_class, 6)
        deltas = deltas[select].view(-1, 6)

        rcnn_reg_loss = 0
        reg_losses = []
        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
            rcnn_reg_loss += l
            reg_losses.append(l.data.item())
    else:
        rcnn_reg_loss = torch.cuda.FloatTensor(1).zero_().sum()

    return rcnn_cls_loss, rcnn_reg_loss, [reg_losses[0], reg_losses[1], reg_losses[2],
                                        reg_losses[3], reg_losses[4], reg_losses[5], confusion_matrix]

def semi_loss(t_logits, s_logits, t_deltas, s_deltas):
    '''
    cls logits: [bg_score, fg_score]
    reg deltas: [bg delta(z,y,x,d,h,w), fg delta]
    '''
    # breakpoint()
    # cls loss
    loss = nn.KLDivLoss(reduction='batchmean')
    t_cls = F.softmax(t_logits, dim=1)
    s_cls = F.log_softmax(s_logits, dim=1)
    cls_loss = loss(s_cls[:,1], t_cls[:,1])
    # pos_num = (s_cls[:,1] > 0.5).sum()
    # cls_loss = loss(s_cls[:,1], t_cls[:,1]) / torch.clamp(pos_num, 1.)

    # reg loss
    t_reg = t_deltas[:,6:]
    s_reg = s_deltas[:,6:]
    reg_loss = torch.zeros(1).cuda()
    if len(t_reg):
        for i in range(6):
            l = F.smooth_l1_loss(s_reg[:, i], t_reg[:, i], beta=(1.0/9.0))
            reg_loss += l
        reg_loss = reg_loss/6

    return cls_loss, reg_loss

if __name__ == '__main__':
    import os
    print( '%s: calling main function ... ' % os.path.basename(__file__))


 
