import os
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from utils.util import crop_with_lobe, get_series, Timer
from dataset.bbox_utils import Crop, Augment, pad2factor, fillter_box
#import SimpleITK as sitk
# import nrrd
import warnings
import json
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

class BboxReader_unlabeled(Dataset):
    def __init__(self, data_dir, set_name, augtype, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = augtype
        self.pad_value = cfg['pad_value']
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.val_filenames = []
        self.set_name = set_name
        self.clip_min = cfg['clip_min']
        self.clip_max = cfg['clip_max']
        self.clip_half = (self.clip_max - self.clip_min)/2
        self.eval_crop = []
        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str)
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)
        elif set_name.endswith('.txt'):
            with open(self.set_name, "r") as f:
                self.filenames = f.read().splitlines()[1:]

        # self.sample_bboxes = []
        # for num, fns in enumerate(self.filenames):
        #     fn = fns.split(',')
        #     with open(os.path.join(fn[0], 'npy', 'lobe_info.txt'), 'r') as f:
        #         lobe_info = f.readlines()[-2:]
        #     Ds, De, Hs, He, Ws, We = crop_with_lobe(*lobe_info, align=1)
        #     patch_series = get_series(Ds, De, Hs, He, Ws, We)

        #     for patch in patch_series:
        #         self.sample_bboxes.append([num, patch])

        self.crop = Crop(cfg)
        self.augment = Augment(cfg)
        # self.datas = {}

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img = False
        if self.mode in ['train', 'val']:
            if idx >= len(self.sample_bboxes):
                is_random_crop = True
                idx = idx % len(self.sample_bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            
            filenum, patch = self.sample_bboxes[idx].copy()              # positive bbox
            filename = self.filenames[filenum]
            imgs, lobe_info = self.load_img(filename,align=1)   # loading images
            imgs = pad2factor(imgs[0], factor=16, pad_value=0.67142)
            imgs = np.expand_dims(imgs, 0)

            # sample = imgs[:, patch[0]:patch[0]+128, patch[1]:patch[1]+128, patch[2]:patch[2]+128]
            sample, target, pos_bbox, coord = self.crop(imgs, patch, [patch], isScale=False, isRand=False)
            sample, target, pos_bbox = self.augment(sample, target, pos_bbox)
            
            bbox = np.array([np.zeros((6))])
            label = np.zeros((1), dtype=np.int16)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape, imgs.shape, lobe_info, patch)

            return [torch.from_numpy(sample.astype(np.float16)), bbox, label]

        if self.mode in ['eval']:
            filename = self.filenames[idx]
            # filename = self.filenames[self.eval_crop[idx][0]]
            
            # imgs = self.load_img(filename)
            imgs, lobe_info = self.load_img(filename, align=32, mode='eval')
            # imgs = self.eval_crop_img(imgs, self.eval_crop[idx])
            # padding
            image = pad2factor(imgs[0], factor=32, pad_value=0.67142)
            image = np.expand_dims(image, 0)

            bboxes = self.sample_bboxes[idx].copy()
            # bboxes = bboxes - [lobe_info[0], lobe_info[2], lobe_info[4], 0, 0, 0]
            # bboxes = self.sample_bboxes[self.eval_crop[idx][0]]
            bboxes = fillter_box(bboxes, imgs.shape[1:])
            # for i in range(3):
            #     bboxes[:, i + 3] = bboxes[:, i + 3] + self.cfg['bbox_border']
            bboxes = np.array(bboxes)
            label = np.ones(len(bboxes), dtype=np.int32)

            input = image.astype(np.float32)
            # return [torch.from_numpy(input), bboxes, label]
            return [torch.from_numpy(input), bboxes, label, lobe_info]

    def __len__(self):
        if self.mode == 'train':
            return int(len(self.sample_bboxes) / (1-self.r_rand))
        elif self.mode =='val':
            return len(self.bboxes)
        else:
            return len(self.filenames)
            # return len(self.eval_crop)

    def eval_crop_img(imgs, crop_config):
        # the last one crop
        if crop_config[1] == crop_config[2]:
            crop_img = imgs[-128:]
        else:
            start_slice = crop_config[1]*64
            crop_img = imgs[start_slice:start_slice+128]
        return crop_img

    def load_img(self, filename, align=16, mode='train'):
        path, dir = filename.split(',')
        img = np.load(os.path.join(path, 'npy', f'{dir}.npy'))
        img = img[np.newaxis,...] # (y, x, z) -> (1, y, x, z)
        ## load lobe info
        with open(os.path.join(path, 'npy', 'lobe_info.txt')) as f:
            # 'z_start,z_end\n'
            # 'y_start,y_end,x_start,x_end'
            lobe_info = f.readlines()[-2:]

        Ds, De, Hs, He, Ws, We = crop_with_lobe(*lobe_info, align=align)
        img = np.clip(img.astype(np.float16), self.clip_min, self.clip_max).transpose(0, 3, 1, 2) # (1, d, h, w)
        images = (img - (self.clip_min + self.clip_half)) / self.clip_half

        # # crop the lobe
        if mode == 'train':
            images = images[:, Ds:De, Hs:He, Ws:We]
        return images, (Ds, De, Hs, He, Ws, We)

    def get_patchs_coordinate(self, net):
        self.sample_bboxes = []
        net.set_mode('eval')
        net.use_rcnn=False
        for i,filename in enumerate(self.filenames):
            imgs, lobe_info = self.load_img(filename,align=32, mode='eval')
            imgs = pad2factor(imgs[0], factor=32, pad_value=0.67142)
            imgs = imgs[np.newaxis, np.newaxis, :]
            # try:
            with autocast(), torch.no_grad():
                net.forward(torch.from_numpy(imgs).cuda(), None, None, lobe_info=[lobe_info])
            # except:
                # breakpoint()
            proposals = net.rpn_proposals.cpu().numpy()
            print(f'\rloading patch {i+1}/{len(self.filenames)}',end='')
            for proposal in proposals:
                coord = proposal[2:]
                self.sample_bboxes.append([i, [int(c)+1 for c in coord]])
        net.use_rcnn=True

    def hu_normalize(self, images):
        images = np.array(images, float)
        images = images - self.clip_min
        images = images / (self.clip_max - self.clip_min)
        return images.astype(np.float32)


if __name__ == '__main__':
    datasets_info = {}
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['stride'] = 4
    datasets_info['bound_size'] = 12
    datasets_info['pad_value'] = 170
    crop = Crop(config=datasets_info)
    sample = np.zeros((512,512,480))
    target = np.array([[1],[1]])
    bboxes = np.array([[[50,50,50,70,70,70],[80,80,80,195,195,150]]])

    s, t, b, c = crop(sample, target, bboxes)