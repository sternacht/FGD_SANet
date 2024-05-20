import numpy as np
import torch
import os
import traceback
import time
# import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from net import *
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from single_config import datasets_info, train_config, test_config, net_config, config
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, crop_boxes2mask_single, \
    npy2submission
import pandas as pd
from evaluationScript.uni_noduleCADEvaluation import noduleCADEvaluation


plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default='eval',
                    help="you want to test or eval")
parser.add_argument('--ckpt', default=test_config['checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--out_dir', default=test_config['out_dir'], type=str, metavar='OUT',
                    help='path to save the results')
parser.add_argument('--test_name', default=datasets_info['test_name'], type=str,
                    help='test set name')
parser.add_argument('--data_dir', default=datasets_info['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--annotation_dir', default=datasets_info['annotation_dir'], type=str, metavar='OUT',
                    help='path to load annotation')
parser.add_argument('--augtype', default=datasets_info['augtype'], type=str, metavar='OUT',
                    help='augment type')


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'eval':
        net = args.net
        initial_checkpoint = args.ckpt
        out_dir = args.out_dir
        test_name = args.test_name
        data_dir = args.data_dir
        annotation_dir = args.annotation_dir
        label_types = config['label_types']
        augtype = args.augtype
        # num_workers = config['num_workers']
        border = config['bbox_border']
        num_workers = 1
        # print(net)

        if 'FGD' in initial_checkpoint:
            config['FGD'] = True
        net = getattr(this_module, net)(config, mode='eval')
        net = net.cuda()
        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            # out_dir = checkpoint['out_dir']
            epoch = checkpoint['epoch']
        else:
            print('No model weight file specified')
            return

        if label_types == 'bbox':
            dataset = BboxReader(data_dir, test_name, augtype, config, mode='eval')
        # else:
        #     dataset = LUNA16BboxReader(data_dir, test_name, augtype, config, mode='eval')
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
        # breakpoint()
        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch)+'')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if os.path.exists(os.path.join(save_dir, 'FROC')):
            create = False
        else:
            os.makedirs(os.path.join(save_dir, 'FROC'))
            create = True

        if create:
            try:
                net.load_state_dict(checkpoint['state_dict'], strict=True)
            except RuntimeError:
                config['FGD'] = False
                net.load_state_dict(checkpoint['state_dict'], strict=True)

            
        eval(net, test_loader, annotation_dir, data_dir, save_dir, border, create_data=create)
    else:
        logging.error('Mode %s is not supported' % (args.mode))

def eval(net, dataset, annotation_dir, data_dir, save_dir=None, border=0, create_data=True):
    net.set_mode('eval')
    # net.use_rcnn = True
    # aps = []
    # dices = []

    print('Total # of eval data %d' % (len(dataset)))
    if create_data:
        print(f'Creating prediction...')
        rpn = []
        detection = []
        ensemble = []
        # for i, (input, truth_bboxes, truth_labels) in tqdm(enumerate(dataset), total=len(dataset), desc='eval'):
        for i, (input, truth_bboxes, truth_labels, lobe_info) in tqdm(enumerate(dataset), total=len(dataset), desc='eval'):
            try:
                input = input.cuda()
                truth_bboxes = np.array(truth_bboxes)
                truth_labels = np.array(truth_labels)
        
                with torch.no_grad():
                    # net.forward(input, truth_bboxes, truth_labels)
                    net.forward(input, truth_bboxes, truth_labels, lobe_info=lobe_info)
                rpns = net.rpn_proposals.cpu().numpy()
                detections = net.detections.cpu().numpy()
                ensembles = net.ensemble_proposals.cpu().numpy()
                # breakpoint()
                if len(rpns):
                    # GT size was expanded by a border to make it larger and easier to detect.
                    # So the true predicted bbox size have to be smaller
                    rpns = rpns[:, 1:] - [0,0,0,0,border, border, border]
                    # rpn.extend(rpns)
                    np.save(os.path.join(save_dir, f'{i}_rpns.npy'), rpns)
        
                if len(detections):
                    detections = detections[:, 1:] - [0,0,0,0,border, border, border]
                    np.save(os.path.join(save_dir, f'{i}_rcnns.npy'), detections)

                if len(ensembles):
                    ensembles = ensembles[:, 1:] - [0,0,0,0,border, border, border]
                    np.save(os.path.join(save_dir, f'{i}_ensembles.npy'), ensembles)

                # Clear gpu memory
                del input, truth_bboxes, truth_labels
                torch.cuda.empty_cache()
        
            except Exception as e:
                del input, truth_bboxes, truth_labels
                torch.cuda.empty_cache()
                traceback.print_exc()
        
                print()
                return
    else:
        print(f'Prediction data has been created')
    print(f'Prediction data creation is done')
    # Generate prediction csv for the use of performing FROC analysis
    # Save both rpn and rcnn results

    rpn_res = []
    rcnn_res = []
    # ensemble_res = []
    for i in range(len(dataset)):
        pid = dataset.dataset.val_filenames[i]
        if os.path.exists(os.path.join(save_dir, f'{i}_rpns.npy')):
            rpns = np.load(os.path.join(save_dir, f'{i}_rpns.npy'))
            # rpns[:, 4] = (rpns[:, 4] + rpns[:, 5] + rpns[:, 6]) / 3
            # rpns = rpns[:, [3, 2, 1, 4, 0]]
            rpns = rpns[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]]*len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, f'{i}_rcnns.npy')):
            rcnns = np.load(os.path.join(save_dir, f'{i}_rcnns.npy'))
            # rcnns[:, 4] = (rcnns[:, 4] + rcnns[:, 5] + rcnns[:, 6]) / 3
            # rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            rcnns = rcnns[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]]*len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        # if os.path.exists(os.path.join(save_dir, f'{i}_ensembles.npy')):
        #     ensembles = np.load(os.path.join(save_dir, f'{i}_ensembles.npy'))
        #     # ensembles[:, 4] = (ensembles[:, 4] + ensembles[:, 5] + ensembles[:, 6]) / 3
        #     # ensembles = ensembles[:, [3, 2, 1, 4, 0]]
        #     ensembles = ensembles[:, [3, 2, 1, 6, 5, 4, 0]]
        #     names = np.array([[pid]]*len(ensembles))
        #     ensemble_res.append(np.concatenate([names, ensembles], axis=1))

    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    # ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['series_id', 'x_center', 'y_center', 'z_center', 'w_mm', 'h_mm', 'd_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    # ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    if not os.path.exists(rpn_submission_path):
        df = pd.DataFrame(rpn_res, columns=col_names)
        df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    # df = pd.DataFrame(ensemble_res, columns=col_names)
    # df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    # if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
    #     os.makedirs(os.path.join(eval_dir, 'ensemble'))

    out = noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))
    #
    # noduleCADEvaluation(annotation_dir, data_dir, dataset.dataset.set_name, ensemble_submission_path,
    #                     os.path.join(eval_dir, 'ensemble'))

    with open(os.path.join(eval_dir, 'rpn', 'fps.txt'),'w') as f:
        f.write(out)
    print


def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]

    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks


if __name__ == '__main__':
    main()