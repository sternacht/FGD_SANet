import os
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from utils.util import crop_with_lobe, pick_rand_neg
from dataset.bbox_utils import Crop, Augment, pad2factor, fillter_box
#import SimpleITK as sitk
# import nrrd
import warnings
import json
import matplotlib.pyplot as plt

class BboxReader_NegNB(Dataset):
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
        labels = []
        self.eval_crop = []
        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str)
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)
        elif set_name.endswith('.txt'):
            with open(self.set_name, "r") as f:
                self.filenames = f.read().splitlines()[1:]
            #self.filenames = ['%05d' % int(i) for i in self.filenames]

        for num, fns in enumerate(self.filenames):
            fn = fns.split(',')
            with open(os.path.join(fn[0], 'npy', 'series_metadata.txt'), 'r') as f:
                nodule_shape_str = f.read().splitlines()[4]
                nh,nw,nd = nodule_shape_str.replace('\n','').split(',')
            with open(os.path.join(fn[0], 'npy', 'lobe_info.txt'), 'r') as f:
                lobe_info = f.readlines()[-2:]
            Ds, De, Hs, He, Ws, We = crop_with_lobe(*lobe_info, align=16)
            fname = os.path.join(fn[0], 'mask', f'{fn[1]}_nodule_count.json')
            with open(fname, 'r') as f:
                annota= json.load(f)
                bboxes = annota['bboxes']
                l = []
                if len(bboxes) > 0:
                    for nodule in bboxes:  # 遍历输入列表
                        top_left = nodule[0]  # 左上角坐标
                        bottom_right = nodule[1]  # 右下角坐标
                        z = (bottom_right[2] + top_left[2]) / 2
                        y = (bottom_right[0] + top_left[0]) / 2
                        x = (bottom_right[1] + top_left[1]) / 2
                        d = (bottom_right[2] - top_left[2]) + 1
                        h = (bottom_right[0] - top_left[0]) + 1
                        w = (bottom_right[1] - top_left[1]) + 1
                        l.append([z, y, x, d, h, w])
                    l = np.array(l)
                    l = l-[Ds, Hs, Ws, 0, 0, 0]
                else:
                    pass
                    # print("No bboxes for %s" % fn)
                l = np.array(l)
                labels.append(l)
                eval_crop_num = math.ceil(int(nd)/64)-1
                for i in range(eval_crop_num):
                    self.eval_crop.append([num, i, eval_crop_num])
                self.eval_crop
            if len(l) > 0:
                self.val_filenames.append(fname)
        # breakpoint()

        self.sample_bboxes = labels
        if self.mode in ['train', 'val', 'eval']:
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        # i-th image (dicom)
                        self.bboxes.append([np.concatenate([[i], t])])
            if self.bboxes == []:
                print()
            self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float32)

        self.crop = Crop(cfg)
        self.augment = Augment(cfg)
        # self.datas = {}

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img = False
        if self.mode in ['train', 'val']:
            if idx >= len(self.bboxes):
                is_random_crop = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            
            pos_bbox = self.bboxes[idx].copy()          # positive bbox
            filename = self.filenames[int(pos_bbox[0])]
            bboxes = self.sample_bboxes[int(pos_bbox[0])].copy()    # all bboxes
            imgs, lobe_info = self.load_img(filename)   # loading images
            imgs = pad2factor(imgs[0], factor=16, pad_value=0.67142)
            imgs = np.expand_dims(imgs, 0)
            # pick a random number to determine pos or neg sample ratio = pos:neg = 3:1
            if np.random.rand() < 0.75:
                # isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, pos_bbox[1:], bboxes, isScale=False, isRand=False)  # crop based on center of nodule
                sample, target, bboxes = self.augment(sample, target, bboxes)   # augment
                bbox = fillter_box(bboxes, self.cfg['crop_size'])  # remove bboxes out of range
                label = np.ones(len(bbox), dtype=np.int16)
            else:
                neg_bbox = pick_rand_neg(bboxes, lobe_info)     # crop random patch as negative sample
                sample = imgs[:, neg_bbox[0]-64:neg_bbox[0]+64, neg_bbox[1]-64:neg_bbox[1]+64, neg_bbox[2]-64:neg_bbox[2]+64]
                bbox = np.array([np.zeros((6))])
                label = np.zeros((1), dtype=np.int16)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape, imgs.shape, lobe_info)

            return [torch.from_numpy(sample.astype(np.float16)), bbox, label]

        if self.mode in ['eval']:
            filename = self.filenames[idx]
            # filename = self.filenames[self.eval_crop[idx][0]]
            
            # imgs = self.load_img(filename)
            imgs, lobe_info = self.load_img(filename, mode='eval')
            # imgs = self.eval_crop_img(imgs, self.eval_crop[idx])
            # padding
            image = pad2factor(imgs[0], pad_value=0.67142)
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
            return int(len(self.bboxes) / (1-self.r_rand))
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
        img = np.load(os.path.join(self.data_dir, '%s\\npy\\%s.npy' % (path, dir)))
        img = img[np.newaxis,...] # (y, x, z) -> (1, y, x, z)
        ## load lobe info
        with open(os.path.join(self.data_dir, '%s\\npy\\lobe_info.txt' %(path))) as f:
            # 'z_start,z_end\n'
            # 'y_start,y_end,x_start,x_end'
            lobe_info = f.readlines()[-2:]

        Ds, De, Hs, He, Ws, We = crop_with_lobe(*lobe_info, align=align)
        # crop the lobe
        if mode == 'train':
            img = img[:, Hs:He, Ws:We, Ds:De]

        img = np.clip(img, self.clip_min, self.clip_max)
        img = img.astype(np.float16)
        img = img.transpose(0, 3, 1, 2) # (1, y, x, z) -> (1, z, y, x)
        images = img - self.clip_min    # 0 ~ max
        images = (images - self.clip_half) / self.clip_half # -1 ~ 1
        return images, (Ds, De, Hs, He, Ws, We)
    
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