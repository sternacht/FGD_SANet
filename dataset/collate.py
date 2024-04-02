import torch
import numpy as np

def train_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)

    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    
    # add padding to bboxes and label to make the same size
    max_label_num = max(label.shape[0] for label in labels)
    if max_label_num > 0:
        label_padded = np.ones((len(labels), max_label_num)) * -1
        bbox_padded = np.ones((len(bboxes), max_label_num, 6)) * -1
        for idx, (label, bbox) in enumerate(zip(labels, bboxes)):
            if label.shape[0] > 0:
                label_padded[idx, :label.shape[0]] = label
                bbox_padded[idx, :bbox.shape[0], :] = bbox
    else:
        label_padded = np.ones((len(labels), 1)) *-1
        bbox_padded = np.ones((len(labels), 1, 6)) *-1

    label_padded = torch.from_numpy(label_padded)
    bbox_padded = torch.from_numpy(bbox_padded)

    return [inputs, bbox_padded, label_padded]


def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    if len(batch[0][1]) == 0:
        bboxes = torch.ones((1, 1, 6)) *-1
        labels = torch.ones((1, 1)) *-1
    else:
        bboxes = [batch[b][1] for b in range(batch_size)]
        labels = [batch[b][2] for b in range(batch_size)]
    images = [batch[b][3] for b in range(batch_size)]

    return [inputs, bboxes, labels, images]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
