import numpy as np
from net.layer.util import box_transform, box_transform_inv, clip_boxes
import itertools
import torch.nn.functional as F
import torch
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap

def make_rpn_windows(f_shape, cfg):
    """
    Generating anchor boxes at each voxel on the feature map,
    the center of the anchor box on each voxel corresponds to center
    on the original input image.

    return
    windows: list of anchor boxes, [z, y, x, d, h, w]
    """
    stride = cfg['stride']
    anchors = np.asarray(cfg['anchors'])
    offset = (float(stride) - 1) / 2
    _, _, D, H, W = f_shape
    oz = np.arange(offset, offset + stride * (D - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (H - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (W - 1) + 1, stride)

    windows = []
    for z, y, x, a in itertools.product(oz, oh , ow , anchors):
        windows.append([z, y, x, a[0], a[1], a[2]])
    windows = np.array(windows)

    return windows

def rpn_nms(cfg, mode, inputs, window, logits_flat, deltas_flat):
    if mode in ['train',]:
        nms_pre_score_threshold = cfg['rpn_train_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_train_nms_overlap_threshold']

    elif mode in ['eval', 'valid', 'test',]:
        nms_pre_score_threshold = cfg['rpn_test_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_test_nms_overlap_threshold']

    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)


    logits = torch.sigmoid(logits_flat).data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()
    batch_size, _, depth, height, width = inputs.size()

    proposals = []
    keeps = np.empty(0)
    offset = 0
    for b in range(batch_size):

        proposal = [np.empty((0, 8),np.float32),]

        ps = logits[b, : , 0].reshape(-1, 1)
        ds = deltas[b, :, :]

        # Only those anchor boxes larger than a pre-defined threshold
        # will be chosen for nms computation
        index = np.where(ps[:, 0] > nms_pre_score_threshold)[0]
        if len(index) > 0:
            p = ps[index]
            d = ds[index]
            w = window[index]
            # 
            # breakpoint()
            box = rpn_decode(w, d, cfg['box_reg_weight'])
            # clip oversized bboxes
            box = clip_boxes(box, inputs.shape[2:])

            output = np.concatenate((p, box),1)

            output = torch.from_numpy(output)
            output, keep = torch_nms(output, nms_overlap_threshold)

            if len(keep):
                # breakpoint()
                keeps = np.append(keeps, index[keep]+offset)
            prop = np.zeros((len(output), 8),np.float32)
            prop[:, 0] = b
            prop[:, 1:8] = output
            
            proposal.append(prop)

        proposal = np.vstack(proposal)
        proposals.append(proposal)
        offset += len(window)
    proposals = np.vstack(proposals)
    # Just in case if there is no proposal, we still return a Tensor,
    # torch.from_numpy() cannot take input with 0 dim
    if len(proposals) != 0:
        proposals = torch.from_numpy(proposals).cuda()
    else:
        proposals = torch.rand([0, 8]).cuda()
    if len(proposals) > 40:
        _,idx = torch.topk(proposals[:,1], 40)
        proposals = proposals[idx]

    return proposals, keeps.astype(np.int32)

def proposal_decoder(cfg, logits_flat, deltas_flat, keeps, window, shape):
    logits = torch.sigmoid(logits_flat).data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()

    proposals = []
    for keep in keeps:
        b = keep//len(window)
        idx = keep%len(window)
        # Only those anchor boxes larger than a pre-defined threshold
        # will be chosen for nms computation
        p = logits[b,idx]
        d = deltas[b,idx]
        w = window[idx]
        # 
        box = rpn_decode(w[np.newaxis,:], d[np.newaxis,:], cfg['box_reg_weight'])
        # clip oversized bboxes
        box = clip_boxes(box, shape[2:])
        
        output = np.concatenate((p[np.newaxis,:], box),1)

        prop = np.zeros((len(output), 8),np.float32)
        prop[:, 0] = b
        prop[:, 1:8] = output
        
        proposals.append(prop)
        
    proposals = np.vstack(proposals)
    proposals = torch.from_numpy(proposals).cuda()
    return proposals

def rpn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)

def rpn_decode(window, delta, weight):
    return box_transform_inv(window, delta, weight)
