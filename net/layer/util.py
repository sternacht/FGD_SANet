import numpy as np
import torch
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes

def box_transform(windows, targets, weight):
    """
    Calculate regression terms, dz, dy, dx, dd, dh, dw
    # windows should equal to # targets
    windows: [num_window, z, y, x, D, H, W]
    targets: [num_target, z, y, x, D, H, W]
    """
    wz, wy, wx, wd, wh, ww = weight
    bz, by, bx = windows[:, 0], windows[:, 1], windows[:, 2]
    bd, bh, bw = windows[:, 3], windows[:, 4], windows[:, 5]
    tz, ty, tx = targets[:, 0], targets[:, 1], targets[:, 2]
    td, th, tw = targets[:, 3], targets[:, 4], targets[:, 5]

    dz = wz * (tz - bz) / bd
    dy = wy * (ty - by) / bh
    dx = wx * (tx - bx) / bw
    dd = wd * np.log(td / bd)
    dh = wh * np.log(th / bh)
    dw = ww * np.log(tw / bw)
    deltas = np.vstack((dz, dy, dx, dd, dh, dw)).transpose()
    return deltas

def box_transform_inv(windows, deltas, weight):
    """
    Apply regression terms to predicted bboxes
    windows: [num_window, z, y, x, D, H, W]
    targets: [num_target, z, y, x, D, H, W]
    """
    if len(windows.shape) == 1:
        windows = windows[np.newaxis,...]
        deltas  = deltas[np.newaxis,...]
    num   = len(windows)
    wz, wy, wx, wd, wh, ww = weight
    predictions = np.zeros((num, 6), dtype=np.float32)
    bz, by, bx = windows[:, 0], windows[:, 1], windows[:, 2]
    bd, bh, bw = windows[:, 3], windows[:, 4], windows[:, 5]
    bz = bz[:, np.newaxis]
    by = by[:, np.newaxis]
    bx = bx[:, np.newaxis]
    bd = bd[:, np.newaxis]
    bh = bh[:, np.newaxis]
    bw = bw[:, np.newaxis]

    dz = deltas[:, 0::6] / wz
    dy = deltas[:, 1::6] / wy
    dx = deltas[:, 2::6] / wx
    dd = deltas[:, 3::6] / wd
    dh = deltas[:, 4::6] / wh
    dw = deltas[:, 5::6] / ww

    z = dz * bd + bz
    y = dy * bh + by
    x = dx * bw + bx
    
    d = np.exp(dd) * bd
    h = np.exp(dh) * bh
    w = np.exp(dw) * bw

    predictions[:, 0::6] = z
    predictions[:, 1::6] = y
    predictions[:, 2::6] = x 
    predictions[:, 3::6] = d
    predictions[:, 4::6] = h
    predictions[:, 5::6] = w

    return predictions

def box_transform_inv_torch(windows, deltas, weight):
    """
    Apply regression terms to predicted bboxes
    windows: [num_window, z, y, x, D, H, W]
    targets: [num_target, z, y, x, D, H, W]
    """
    if len(windows.shape) == 1:
        windows = windows.unsqueeze(0)
        deltas = deltas.unsqueeze(0)
    num = windows.size(0)
    wz, wy, wx, wd, wh, ww = weight
    predictions = torch.zeros((num, 6), dtype=torch.float32)
    bz, by, bx = windows[:, 0], windows[:, 1], windows[:, 2]
    bd, bh, bw = windows[:, 3], windows[:, 4], windows[:, 5]
    bz = bz.unsqueeze(1)
    by = by.unsqueeze(1)
    bx = bx.unsqueeze(1)
    bd = bd.unsqueeze(1)
    bh = bh.unsqueeze(1)
    bw = bw.unsqueeze(1)

    dz = deltas[:, 0::6] / wz
    dy = deltas[:, 1::6] / wy
    dx = deltas[:, 2::6] / wx
    dd = deltas[:, 3::6] / wd
    dh = deltas[:, 4::6] / wh
    dw = deltas[:, 5::6] / ww

    z = dz * bd + bz
    y = dy * bh + by
    x = dx * bw + bx
    
    d = torch.exp(dd) * bd
    h = torch.exp(dh) * bh
    w = torch.exp(dw) * bw

    predictions[:, 0::6] = z
    predictions[:, 1::6] = y
    predictions[:, 2::6] = x 
    predictions[:, 3::6] = d
    predictions[:, 4::6] = h
    predictions[:, 5::6] = w

    return predictions

def clip_boxes(boxes, img_size):
    """
    clip boxes outside the image, all box follows [p, z, y, x, d, h, w]
    """
    depth, height, width = img_size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, depth)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, depth)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 4] = np.clip(boxes[:, 4], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 5] = np.clip(boxes[:, 5], 0, width)

    return boxes

def proposal_jittering(proposals, gamma=0.2, times=3):
    '''
    random jitter proposals
    gamma: jittering ratio, 0.5>gamma>=0
    '''
    proposals = proposals.reshape(-1, 8)
    jittered_proposals = []
    for proposal in proposals:
        jittered_proposals.append(proposal)
        r = np.random.rand(times,6)
        r[:,:3] = r[:,:3]* (2*gamma) - gamma
        r[:,3:] = r[:,3:]* (4*gamma) + (1-2*gamma) 
        for i in range(3):
            z = proposal[2] + proposal[5]*r[i][0]
            y = proposal[3] + proposal[6]*r[i][1]
            x = proposal[4] + proposal[7]*r[i][2]
            d = proposal[5] * r[i][3]
            h = proposal[6] * r[i][4]
            w = proposal[7] * r[i][5]
            jittered_proposals.append([proposal[0], proposal[1], z,y,x,d,h,w])
    return np.array(jittered_proposals)

def proposal_preprocess(proposals, inputs_size):
    proposal = proposals[:, [0, 2, 3, 4, 5, 6, 7]]                               # pick proposal=[bs,z,y,x,d,h,w]
    proposal[:, 1:] = center_box_to_coord_box(proposal[:, 1:])                   # transform [bs,z,y,x,d,h,w] to [bs,z_start,..., z_end,...]
    proposal = proposal.astype(np.int64)                                         # make it integer
    proposal[:, 1:] = ext2factor(proposal[:, 1:], 4)                             # align to 4
    proposal[:, 1:] = clip_boxes(proposal[:, 1:], inputs_size[2:])               # clip to make boxes in range of image
    return proposal

def clip_delta_variance(deltas, proposals, threshold):
    # breakpoint()
    deltas = deltas.reshape(-1,4,2,3)
    var = np.std(deltas, axis=1)            # compute std of delta though it's called varience...
    size = proposals[:,5:][:, np.newaxis]
    mean_var = np.mean(var/size, axis=(1,2))
    keeps = np.where(mean_var<=threshold)
    return keeps

def swap(arr):
    while True:
        axisorder = torch.randperm(3)
        if not torch.all(axisorder == torch.tensor([0, 1, 2])):
            break
    axisorder = torch.cat([torch.tensor([0,1]),axisorder+2], dim=0)
    arr = arr.permute(*axisorder)
    unswap = [0, 0, 0, 0, 0, 5, 6]
    for i in range(5):
        unswap[axisorder[i]] = i
    return arr, torch.tensor(unswap)

def flip(arr):
    flipid = np.where([0,0,np.random.randint(2),np.random.randint(2),np.random.randint(2)])[0]
    if flipid.sum() == 0:
        flipid = [np.random.randint(2,5)]
    
    arr = torch.flip(arr, dims=tuple(flipid))

    return arr, flipid

def strong_augment(inputs):
    inputs, unswapax = swap(inputs)
    inputs, unflipid = flip(inputs)
    return inputs, unswapax, unflipid

def unswap(arr, unswap):
    return arr.permute(*unswap)

def unflip(arr, unflip):
    return torch.flip(arr, dims=tuple(unflip)).contiguous()

def strong_recover(inputs, unflipid, unswapax):
    b, s, n = inputs.shape
    inputs = inputs.contiguous().view(b, 1, 32, 32, 32, 6, n)
    outputs = unflip(inputs,  unflipid)
    outputs = unswap(outputs, unswapax)
    outputs = outputs.contiguous().view(b, s, n)
    return outputs