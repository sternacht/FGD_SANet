import os
import numpy as np
import torch
import random


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Preprocessing using preserved HU in dilated part of mask
BASE = '/data/' # make sure you have the ending '/'

test_config = {
    'dataset_train': 'pn9_sanet',
    'dataset_test': 'pn9_sanet',
    'load_epoch': '015',

}
# dataset = 'pn9_sanet' # dataset_test
dataset = 'ME_LDCT'
datasets_info = {}
if dataset == 'luna16_sgda_aug':
    datasets_info['dataset'] = 'luna16_sgda_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv' # test
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_sgda_aug':
    datasets_info['dataset'] = 'russia_diannei_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_sgda_aug':
    datasets_info['dataset'] = 'tianchi_diannei_aug'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/sgda_split/train.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune0':
    datasets_info['dataset'] = 'luna16_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune0':
    datasets_info['dataset'] = 'russia_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune0':
    datasets_info['dataset'] = 'tianchi_finetune0'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train0.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune1':
    datasets_info['dataset'] = 'luna16_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune1':
    datasets_info['dataset'] = 'russia_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune1':
    datasets_info['dataset'] = 'tianchi_finetune1'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train1.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune2':
    datasets_info['dataset'] = 'luna16_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune2':
    datasets_info['dataset'] = 'russia_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune2':
    datasets_info['dataset'] = 'tianchi_finetune2'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train2.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'luna16_finetune3':
    datasets_info['dataset'] = 'luna16_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/luna16/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/luna16/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/luna16/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/luna16/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/luna16/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8 #8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'russia_finetune3':
    datasets_info['dataset'] = 'russia_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/russia/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/russia/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/russia/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/russia/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/russia/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'tianchi_finetune3':
    datasets_info['dataset'] = 'tianchi_finetune3'
    datasets_info['train_list'] = ['/home/xurui/data0/tianchi/finetune_split/train3.csv']
    datasets_info['val_list'] = ['/home/xurui/data0/tianchi/sgda_split/val.csv']
    datasets_info['test_name'] = '/home/xurui/data0/tianchi/sgda_split/test.csv'
    datasets_info['data_dir'] = '/home/xurui/data0/tianchi/'
    datasets_info['annotation_dir'] = '/home/xurui/data0/tianchi/annotations.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'pn9_sanet':
    datasets_info['dataset'] = 'pn9_sanet'
    datasets_info['train_list'] = ['/data0/pn9/split_full_with_nodule_9classes/train.txt']
    datasets_info['val_list'] = ['/data0/pn9/split_full_with_nodule_9classes/val.txt']
    datasets_info['test_name'] = '/data0/pn9/split_full_with_nodule_9classes/test.txt'
    datasets_info['data_dir'] = '/data0/pn9/'
    datasets_info['annotation_dir'] = '/data0/pn9/annotations.csv'
    datasets_info['BATCH_SIZE'] = 16
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 8
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': False, 'rotate': False, 'scale': True, 'swap': False}
elif dataset == 'ME_LDCT':
    datasets_info['dataset'] = 'ME_LDCT'
    datasets_info['train_list'] = [r'D:\master\code\Lung_Nodule\FL_lung_nodule_datasplit_ME_LDCT\client0_train.txt']
    datasets_info['val_list'] = [r'D:\master\code\Lung_Nodule\FL_lung_nodule_datasplit_ME_LDCT\client0_val.txt']
    datasets_info['test_name'] = r'D:\master\code\Lung_Nodule\FL_lung_nodule_datasplit_ME_LDCT\client0_test.txt'
    datasets_info['data_dir'] = r'D:\master\code\Lung_Nodule\dataset\ME_dataset'
    datasets_info['clip_max'] = 400
    datasets_info['clip_min'] = -1000
    datasets_info['annotation_dir'] = r'D:\master\code\Lung_Nodule\FL_lung_nodule_datasplit_ME_LDCT\client0_test_annotation.csv'
    datasets_info['BATCH_SIZE'] = 4
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 4
    datasets_info['pad_value'] = 170
    datasets_info['augtype'] = {'flip': True, 'rotate': False, 'scale': True, 'swap': False}

def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors

# bases = [5,10,20,30,50]
# aspect_ratios = [[1,1,1]]

bases = [6, 12, 24]
aspect_ratios = [[1, 1, 1],
                 [1.5, 1.5, 1]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    'crop_size': datasets_info['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': ['09184', '05181','07121','00805','08832'],
    # 'blacklist': [],

    # 'augtype': {'flip': False, 'rotate': False, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    'rpn_fg_threshold': 0.5,
    # false positive reduction network configuration
    'num_class': len(datasets_info['roi_names']) + 1,
    'rcnn_crop_size': (7,7,7), # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.],
    
    'FGD': False, # using FGD to train student model
    'teacher': r'D:\master\code\LSSANet-main\SANet_L3S2_results\custom\warmupLr0.005_Ercnn80_Bs4_OHEM\model\165.ckpt',
}

def lr_shedule(epoch, init_lr=1e-1, total=200):
    # warmup in 20 epochs
    if epoch == 0:
        lr = init_lr/100
    elif epoch < total * 0.1:
        lr = init_lr * (epoch/(total*0.1))

    elif epoch <= total * 0.25:
        lr = init_lr
    elif epoch <= total * 0.6:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'MsaNet', # MsaNet, SANet_L3S2
    'num_groups': 4,
    'batch_size': datasets_info['BATCH_SIZE'],

    'lr_schedule': lr_shedule,
    'optimizer': 'AdamW',
    'momentum': 0.9,
    'weight_decay': 1e-5,

    'epochs': 120, #200 #400
    'epoch_save': 3,
    'epoch_rcnn': 100, #20 #47
    'num_workers': 4, #30

    'hard_example_solution': 'OHEM',# 'focal' or 'OHEM'
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.005
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.0005
elif train_config['optimizer'] == 'AdamW':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

train_config['RESULTS_DIR'] = os.path.join(r'D:\master\code\LSSANet-main\{}_results'.format(train_config['net']),
                                  datasets_info['dataset'])
# train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], f"test")
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], f"{train_config['optimizer']}{train_config['init_lr']}_Ercnn{train_config['epoch_rcnn']}_Bs{datasets_info['BATCH_SIZE']}_{train_config['hard_example_solution']}_wd1e-5_fullrpn")
if net_config['FGD']:
    train_config['out_dir'] += '_FGD'
train_config['initial_checkpoint'] = None
# train_config['initial_checkpoint'] = r'D:\master\code\LSSANet-main\SANet_L3S2_results\custom\warmupLr0.01_Ercnn10_Bs8\model\061.ckpt'

# test_config['out_dir'] = 'D:/master/code/LSSANet-main/{}_results/{}/cross_val_test_{}/{}/'.format(
#     train_config['net'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['dataset_test'])
out_dir = r'D:\master\code\LSSANet-main\MsaNet_results\ME_LDCT\AdamW0.001_Ercnn100_Bs4_focal_wd1e-5_fullrpn'

test_config['out_dir'] = out_dir
# test_config['checkpoint'] = 'D:/master/code/LSSANet-main/{}_results/{}/cross_val_test_{}/model/{}.ckpt'.format(
#     train_config['net'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['load_epoch'])
test_config['checkpoint'] = rf'{out_dir}\model\018.ckpt'
test_config['valid_checkpoint'] = r'D:\master\code\LSSANet-main\SANet_L3S2_results\custom\warmupLr0.005_Ercnn80_Bs4_OHEM_FGD\model\079.ckpt'
config = dict(datasets_info, **net_config)
config = dict(config, **train_config)
config = dict(config, **test_config)