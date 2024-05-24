from net.sanet import SANet
from net.sanet_l3s2 import SANet_L3S2
from net.MSANet import MsaNet
from net.MSANet_reduce import MsaNet_R
from net.layer import *
import time
from dataset.collate import train_collate, test_collate, eval_collate, train_with_neg_collate
from dataset.bbox_reader import BboxReader
from dataset.bbox_reader_neg import BboxReader_Neg
from dataset.bbox_reader_neg_nodulebased import BboxReader_NegNB
from dataset.bbox_reader_unlabeled import BboxReader_unlabeled
from utils.util import Logger
from single_config import *
import pprint
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import data_parallel
import torch
import numpy as np
import argparse
import os
import sys
from termcolor import cprint
from colorama import Fore
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-list', default=datasets_info['l_train_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--ul_train-set-list', default=datasets_info['ul_train_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set-list', default=datasets_info['val_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=datasets_info['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batchsize-scale', '-lbs', default=train_config['batchsize_scale'], type=int,
                    help="Train model with larger batch size, scaled by this param.")


def main():
    # Load training configuration
    args = parser.parse_args()
    torch.backends.cudnn.benchmark=True

    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
    train_set_list = args.train_set_list
    val_set_list = args.val_set_list
    # lr_schdule = train_config['lr_schedule']
    data_dir = args.data_dir
    label_type = config['label_types']
    augtype = config['augtype']
    batchsize_scale = args.batchsize_scale
    ul_train_set = args.ul_train_set_list
    
    # Initilize network
    net = getattr(this_module, net)(config, hes=train_config['hard_example_solution'])
    print(f'CUDA:{torch.cuda.is_available()}')
    net = net.cuda()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    optimizer = getattr(torch.optim, optimizer)
    if isinstance(optimizer, torch.optim.SGD):
        optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay)
    # lr_schduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45, 90, 100], gamma=0.1)
    lr_schduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1, eta_min=train_config['init_lr']/100)
    
    # def warmup_fn(epoch, warmup_epochs=10):
    #     if epoch < warmup_epochs:
    #         return epoch / warmup_epochs
    #     else:
    #         return 1
    # warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_fn)
    # cosine_annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 - 10)
    # lr_schduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler,cosine_annealing_scheduler], milestones=[10])
    
    scaler = GradScaler()

    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        # if not ('model.ckpt' in initial_checkpoint):
            # start_epoch = checkpoint['epoch']

        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            net.load_state_dict(state, strict=True)
            if not ('model.ckpt' in initial_checkpoint):
                optimizer.load_state_dict(checkpoint['optimizer'])
            for i in range(start_epoch):
                lr_schduler.step()
            if start_epoch >= epoch_rcnn:
                net.use_rcnn = True
        except:
            print('Load something failed!')
            traceback.print_exc()
           
    teacher_net = SANet(config, mode='train')
    teacher_net.load_state_dict(torch.load(config['teacher'])['state_dict'])
    teacher_net = teacher_net.cuda()
    teacher_net.eval()

    if label_type == 'bbox':
        unlabeled_train_dataset = BboxReader_unlabeled(data_dir, ul_train_set, augtype, config, mode='train')
        labeled_train_dataset = BboxReader_NegNB(data_dir, train_set_list, augtype, config, mode='train')
        val_dataset = BboxReader(data_dir, val_set_list, augtype, config, mode='val')
    unlabeled_train_dataset.get_patchs_coordinate(teacher_net)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate)
    
    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(labeled_train_loader)+len(unlabeled_train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)
    reload_path_count = 1
    for i in range(start_epoch, epochs + 1):
        # learning rate schedule
        lr = train_config['init_lr']

        if i == epoch_rcnn:
            net.use_rcnn = True
            # net.freeze_featurenet_rpn(grad=True)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = train_config['init_lr']

        print('[epoch %d/%d, lr %f, use_rcnn: %r]' % (i, epochs, optimizer.param_groups[0]['lr'], net.use_rcnn))
        train(net, teacher_net, unlabeled_train_loader, labeled_train_loader, optimizer, i, 
              train_writer, scaler=scaler, batchsize_scale=batchsize_scale)
        lr_schduler.step()

        state_dict = net.state_dict()
        for key in state_dict.keys():
            if 'teacher' not in key:
                state_dict[key] = state_dict[key].cpu()

        # breakpoint()
        if i % epoch_save == 0 and i>0:
            validate(net, val_loader, i, val_writer)
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))
        torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, 'last.ckpt'))
        
        for param1, param2 in zip(teacher_net.parameters(), net.parameters()):
            param1.data = 0.999*param1.data + 0.001*param2.data

        if (reload_path_count)%20 == 0:
            unlabeled_train_dataset.get_patchs_coordinate(teacher_net)
            unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)
        reload_path_count += 1
    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, teacher, ul_train_loader, l_train_loader, optimizer, epoch, writer, scaler=None, batchsize_scale=1):
    net.set_mode('train')
    teacher.set_mode('train')
    s = time.time()
    rpn_cls_losses, rpn_reg_losses = [], []
    rcnn_cls_losses, rcnn_reg_losses = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []
    att_losses = []
    fg_losses = []
    bg_losses = []
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    labeled_data = iter(l_train_loader)
    with tqdm(total=len(ul_train_loader)) as pbar:
        optimizer.zero_grad()
        for j, (inputs, truth_box, truth_label) in enumerate(ul_train_loader):
            # first train unlabeled data
            truth_box = np.array(truth_box)
            truth_label = np.array(truth_label)

            semi_cls_loss = torch.zeros(1).to(device)
            semi_reg_loss = torch.zeros(1).to(device)

            with autocast():
                inputs = inputs.to(device, non_blocking=True)
                with torch.no_grad():
                    teacher(inputs, truth_box, truth_label)
                # breakpoint()
                t_rpn_proposals, keeps = rpn_nms(config, 'train', inputs, 
                                                 teacher.rpn_window,
                                                 teacher.rpn_logits_flat,
                                                 teacher.rpn_deltas_flat)
                t_rpn_proposals = t_rpn_proposals.detach()
                if len(keeps):
                    # cls_idx = torch.where(t_rpn_proposals[:,1] >= 0.5)
                    # reg_idx = torch.where(t_rpn_proposals[:,1] >= 0.5)
                    rcnn_cand_idx = torch.where(t_rpn_proposals[:,1] >= 0.5)
                    
                    t_features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in teacher.features]

                    # if len(cls_idx[0]):
                    #     t_cls_proposal = proposal_preprocess(t_rpn_proposals[cls_idx].cpu().numpy(), inputs.shape)
                    #     with torch.no_grad():
                    #         t_cls_rcnn_crops = data_parallel(teacher.rcnn_crop, (t_features, inputs, torch.from_numpy(t_cls_proposal).cuda()))
                    #         t_cls_rcnn_logits, _ = data_parallel(teacher.rcnn_head, t_cls_rcnn_crops)

                    # if len(reg_idx[0]):
                    #     t_reg_proposal_candidate = proposal_jittering(t_rpn_proposals[reg_idx].cpu().numpy(), gamma=0.3)
                    #     t_reg_proposal = proposal_preprocess(t_reg_proposal_candidate, inputs.shape)
                    #     with torch.no_grad():
                    #         t_reg_rcnn_crops = data_parallel(teacher.rcnn_crop, (t_features, inputs, torch.from_numpy(t_reg_proposal).cuda()))
                    #         _, t_reg_rcnn_deltas = data_parallel(teacher.rcnn_head, t_reg_rcnn_crops)
                    #     cord_box_deltas = center_box_to_coord_box(box_transform_inv(t_reg_proposal_candidate[:,2:],
                    #                     t_reg_rcnn_deltas.cpu().detach().numpy()[:,6:], config['box_reg_weight']))
                    #     reg_keeps = clip_delta_variance(cord_box_deltas, t_rpn_proposals[reg_idx].cpu().numpy(), threshold=0.2)

                    if len(rcnn_cand_idx[0]):
                        t_proposal_candidate = proposal_jittering(t_rpn_proposals[rcnn_cand_idx].cpu().numpy(), gamma=0.2)
                        t_proposal = proposal_preprocess(t_proposal_candidate, inputs.shape)
                        with torch.no_grad():
                            t_rcnn_crops = data_parallel(teacher.rcnn_crop, (t_features, inputs, torch.from_numpy(t_proposal).cuda()))
                            t_cls_rcnn_logits, t_reg_rcnn_deltas = data_parallel(teacher.rcnn_head, t_rcnn_crops)
                        t_cls_rcnn_logits = t_cls_rcnn_logits.reshape(len(rcnn_cand_idx[0]),4,2)[:,0,...]
                        cord_box_deltas = center_box_to_coord_box(box_transform_inv(t_proposal_candidate[:,2:],
                                        t_reg_rcnn_deltas.cpu().detach().numpy()[:,6:], config['box_reg_weight']))
                        reg_keeps = clip_delta_variance(cord_box_deltas, t_rpn_proposals[rcnn_cand_idx].cpu().numpy(), threshold=0.1)

                    inputs, unswapax, unflipid = strong_augment(inputs)
                    net(inputs, truth_box, truth_label)

                    s_features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in net.features]
                    s_rpn_logits_flat = strong_recover(net.rpn_logits_flat, unflipid, unswapax)
                    s_rpn_deltas_flat = strong_recover(net.rpn_deltas_flat, unflipid, unswapax)
                    # breakpoint()
                    s_rpn_proposals = proposal_decoder(config, 
                                                    s_rpn_logits_flat,
                                                    s_rpn_deltas_flat,
                                                    keeps, 
                                                    net.rpn_window, 
                                                    inputs.shape)
                    if len(s_rpn_proposals):
                        s_proposals = proposal_preprocess(s_rpn_proposals.detach().cpu().numpy(), inputs.shape)
                        s_rcnn_crops = data_parallel(net.rcnn_crop, (s_features, inputs, torch.from_numpy(s_proposals).cuda()))
                        s_rcnn_logits, s_rcnn_deltas = data_parallel(net.rcnn_head, s_rcnn_crops)
                        semi_cls_loss, semi_reg_loss = semi_loss(t_cls_rcnn_logits, s_rcnn_logits[rcnn_cand_idx],
                                    t_reg_rcnn_deltas[reg_keeps], s_rcnn_deltas[rcnn_cand_idx][reg_keeps])
                                
            # then train labeled data
            try:
                inputs, truth_box, truth_label = next(labeled_data)
            except:
                labeled_data = iter(l_train_loader)
                inputs, truth_box, truth_label = next(labeled_data)
            truth_box = np.array(truth_box)
            truth_label = np.array(truth_label)
            with autocast():
                inputs = inputs.to(device, non_blocking=True)
                inputs, unswapax, unflipid = strong_augment(inputs)
                net(inputs, truth_box, truth_label)
                # breakpoint()
                rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_target_weights = make_rpn_target(config, 
                        'train', inputs, net.rpn_window, truth_box, truth_label)
    
                rpn_logits_flat = strong_recover(net.rpn_logits_flat, unflipid, unswapax)
                rpn_deltas_flat = strong_recover(net.rpn_deltas_flat, unflipid, unswapax)
                rpn_cls_loss, rpn_reg_loss, rpn_stat = rpn_loss(rpn_logits_flat, rpn_deltas_flat, rpn_labels,
                        rpn_label_weights, rpn_targets, rpn_target_weights, config, 'train', 'fOHEM')

                # breakpoint()
                rpn_proposals,_ = rpn_nms(config, 'train', inputs, net.rpn_window, rpn_logits_flat, rpn_deltas_flat)
                rpn_proposals, rcnn_labels, rcnn_assigns, rcnn_targets = make_rcnn_target(config, 'train', inputs, 
                        rpn_proposals, truth_box, truth_label)
                proposal = rpn_proposals[:, [0, 2, 3, 4, 5, 6, 7]].cpu().numpy().copy() # pick proposal=[bs,z,y,x,d,h,w]
                proposal[:, 1:] = center_box_to_coord_box(proposal[:, 1:])              # transform [bs,z,y,x,d,h,w] to [bs,z_start,..., z_end,...]
                proposal = proposal.astype(np.int64)                                    # make it integer
                proposal[:, 1:] = ext2factor(proposal[:, 1:], 4)                        # align to 4
                proposal[:, 1:] = clip_boxes(proposal[:, 1:], inputs.shape[2:])         # clip to make boxes in range of image
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in net.features]
                rcnn_crops = data_parallel(net.rcnn_crop, (features, inputs, torch.from_numpy(proposal).cuda()))
                
                rcnn_logits, rcnn_deltas = data_parallel(net.rcnn_head, rcnn_crops)
                rcnn_cls_loss, rcnn_reg_loss, rcnn_stat = rcnn_loss(rcnn_logits, rcnn_deltas, rcnn_labels, rcnn_targets)
            
                loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss + semi_cls_loss + semi_reg_loss

            loss = loss/batchsize_scale
            # print(loss.data)
            if scaler:
            # mixed precision loss backward
                scaler.scale(loss).backward()
                if (j+1)%batchsize_scale == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            # normal loss backward
            else:
                loss.backward()
                if (j+1)%batchsize_scale == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
            total_loss.append(loss.cpu().data.item())
            rpn_cls_losses.append(rpn_cls_loss.cpu().data.item())
            rpn_reg_losses.append(rpn_reg_loss.cpu().data.item())
            rcnn_cls_losses.append(rcnn_cls_loss.cpu().data.item())
            rcnn_reg_losses.append(rcnn_reg_loss.cpu().data.item())
            rpn_stats.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
            # if net_config["FGD"]:
            #     att_losses.append(att_loss.cpu().data.item())
            #     fg_losses.append(fg_loss.cpu().data.item())
            #     bg_losses.append(bg_loss.cpu().data.item())

            pbar.set_description(f'loss:{np.array(total_loss).mean():.4f}')
            pbar.update(1)

            rcnn_stats.append(rcnn_stat)

            # del inputs, truth_box, truth_label
            # del rpn_proposals
            # del loss, rcnn_cls_loss, rcnn_reg_loss, rpn_cls_loss, rpn_reg_loss, semi_cls_loss, semi_reg_loss
            # del net.rpn_logits_flat, net.rpn_deltas_flat
            # del teacher.rpn_logits_flat, teacher.rpn_deltas_flat
            # del rcnn_stat, rpn_stat

            # if net.use_rcnn:
            #     del net.rcnn_logits, net.rcnn_deltas
            if net_config["FGD"]:
                del fgd_loss
            torch.cuda.empty_cache()

    rpn_stats = np.asarray(rpn_stats, np.float32)
    
    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    if config['FGD']:
        print(f'{Fore.RED}rpn_cls {np.average(rpn_cls_losses)}, rpn_reg {np.average(rpn_reg_losses)}, fg {np.average(fg_losses)}'
              f', bg {np.average(bg_losses)}, att {np.average(att_losses)}{Fore.RESET}')
    else:
        print(f'{Fore.RED}rpn_cls {np.average(rpn_cls_losses)}, rpn_reg {np.average(rpn_reg_losses)}{Fore.RESET}')

    TP = np.sum(rpn_stats[:, 0])
    recall = 100.0 * TP / np.sum(rpn_stats[:, 1])  #TP/(TP+FN)->TP/total pos
    precision = 100.0 * TP / (TP + np.sum(rpn_stats[:, 3])-np.sum(rpn_stats[:, 2])) #TP/(TP+FP)
    F1_score = 2*recall*precision/(recall+precision)
    print(f'{Fore.MAGENTA}rpn_stats: recall(tpr) {recall}, '
          f'tnr {100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3])}, total pos {int(np.sum(rpn_stats[:, 1]))}, '
          f'total neg {int(np.sum(rpn_stats[:, 3]))}, reg {np.mean(rpn_stats[:, 4]):.4f}, {np.mean(rpn_stats[:, 5]):.4f}, '
          f'{np.mean(rpn_stats[:, 6]):.4f}, {np.mean(rpn_stats[:, 7]):.4f}, '
          f'{np.mean(rpn_stats[:, 8]):.4f}, {np.mean(rpn_stats[:, 9]):.4f}{Fore.RESET}')
    # Write to tensorboard
    writer.add_scalar('tpr', recall, epoch)
    writer.add_scalar('tnr', (100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3])), epoch)
    # writer.add_scalar('F1_score',F1_socre, epoch)

    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_losses), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_losses), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_losses), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_losses), epoch)

    if net_config['FGD']:
        writer.add_scalar('att_loss', np.average(att_losses), epoch)
        writer.add_scalar('fg_loss', np.average(fg_losses), epoch)
        writer.add_scalar('bg_loss', np.average(bg_losses), epoch)
    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)

        cprint('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])), 'yellow')
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    

def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    s = time.time()

    with tqdm(total=len(val_loader)) as pbar:
        for j, (inputs, truth_box, truth_label) in enumerate(val_loader):
            with torch.no_grad():
                inputs = inputs.cuda()
                truth_box = np.array(truth_box)
                truth_label = np.array(truth_label)

                # with autocast():
                net(inputs, truth_box, truth_label)
                loss, rpn_stat, rcnn_stat = net.loss()
            
            total_loss.append(loss.cpu().data.item())
            rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
            rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
            rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
            rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())

            pbar.set_description(f'loss:{np.array(total_loss).mean():.4f}')
            pbar.update(1)

            rpn_stats.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
            if net.use_rcnn:
                rcnn_stats.append(rcnn_stat)
                del rcnn_stat

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    print(f'{Fore.GREEN}rpn_cls {np.average(rpn_cls_loss)}, rpn_reg {np.average(rpn_reg_loss)}{Fore.RESET}')
    
    TP = np.sum(rpn_stats[:, 0])
    recall = 100.0 * TP / np.sum(rpn_stats[:, 1])  #TP/(TP+FN)->TP/total pos
    precision = 100.0 * TP / (TP + np.sum(rpn_stats[:, 3])-np.sum(rpn_stats[:, 2])) #TP/(TP+FP)
    F1_score = 2*recall*precision/(recall+precision)
    print(f'{Fore.CYAN}rpn_stats: recall(tpr) {recall}, '
          f'tnr {100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3])}, total pos {int(np.sum(rpn_stats[:, 1]))}, '
          f'total neg {int(np.sum(rpn_stats[:, 3]))}, reg {np.mean(rpn_stats[:, 4]):.4f}, {np.mean(rpn_stats[:, 5]):.4f}, '
          f'{np.mean(rpn_stats[:, 6]):.4f}, {np.mean(rpn_stats[:, 7]):.4f}, '
          f'{np.mean(rpn_stats[:, 8]):.4f}, {np.mean(rpn_stats[:, 9]):.4f}{Fore.RESET}')

    # Write to tensorboard
    writer.add_scalar('tpr', recall, epoch)
    writer.add_scalar('tnr', (100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3])), epoch)
    writer.add_scalar('F1_score', F1_score, epoch)
    writer.add_scalar('precision', precision, epoch)
    
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)
        cprint('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])),color='blue')
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    del inputs, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
    del rpn_stat

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()

def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))
        

if __name__ == '__main__':
    main()



