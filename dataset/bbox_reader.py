import os
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from utils.util import crop_with_lobe
#import SimpleITK as sitk
# import nrrd
import warnings
import json
import matplotlib.pyplot as plt

class BboxReader(Dataset):
    def __init__(self, data_dir, set_name, augtype, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = augtype
        self.pad_value = cfg['pad_value']
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.val_filenames = []
        # self.blacklist = cfg['blacklist']
        self.blacklist = ['RLADD02000096137_RLSDD02000096902', 'RLADD02000022021_RLSDD02000020110', 
                          '09184', '05181', '07121', '00805', '08832', '00036', '00161']
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
        if self.mode != 'test':
            self.filenames = [f for f in self.filenames if (f not in self.blacklist)]

        for num, fns in enumerate(self.filenames):
            fn = fns.split(',')
            fname = os.path.join(fn[0], 'mask', f'{fn[1]}_nodule_count.json')
            f_noduleinfo = os.path.join(fn[0], 'npy', 'series_metadata.txt')
            with open(f_noduleinfo, 'r') as f:
                nodule_shape_str = f.read().splitlines()[4]
                nh,nw,nd = nodule_shape_str.replace('\n','').split(',')
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
                else:
                    pass
                    # print("No bboxes for %s" % fn)
                l = np.array(l)
                #l = fillter_box(l, [512, 512, 512])
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
            if not is_random_img:
                # if idx in self.datas.keys():
                #     sample, bboxes = self.datas[idx]
                # else:
                bbox = self.bboxes[idx].copy()
                filename = self.filenames[int(bbox[0])]
                # imgs = self.load_img(filename)
                imgs, lobe_info = self.load_img(filename)
                bboxes = self.sample_bboxes[int(bbox[0])].copy()
                bbox = bbox - [0, lobe_info[0], lobe_info[2], lobe_info[4], 0, 0, 0]
                bboxes = bboxes - [lobe_info[0], lobe_info[2], lobe_info[4], 0, 0, 0]
                # bboxes = fillter_box(bboxes, imgs.shape[1:])
                isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale=False, isRand=False)
                # if self.mode == 'train' and not is_random_crop:
                sample, target, bboxes = augment(sample, target, bboxes, do_flip=self.augtype['flip'],
                                                do_rotate=self.augtype['rotate'], do_swap=self.augtype['swap'])
                # self.datas[idx] = [sample, bboxes]
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                # imgs = self.load_img(filename)
                imgs, lobe_info = self.load_img(filename)
                bboxes = self.sample_bboxes[randimid].copy()
                bboxes = bboxes - [lobe_info[0], lobe_info[2], lobe_info[4], 0, 0, 0]
                bboxes = fillter_box(bboxes, imgs.shape[1:])
                isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape, imgs.shape, lobe_info)

            # Normalize
            # sample = self.hu_normalize(sample)
            # draw_all_bbox(sample,target,filename)
            sample = sample.astype(np.float32)
            bboxes = fillter_box(bboxes, self.cfg['crop_size'])
            label = np.ones(len(bboxes), dtype=np.int32)
            # print(bboxes.shape)
            if len(bboxes.shape) != 1:
                for i in range(3):
                    bboxes[:, i+3] = bboxes[:, i+3] + self.cfg['bbox_border']

            return [torch.from_numpy(sample), bboxes, label]

        if self.mode in ['eval']:
            filename = self.filenames[idx]
            # filename = self.filenames[self.eval_crop[idx][0]]
            
            # imgs = self.load_img(filename)
            imgs, lobe_info = self.load_img(filename, align=32, mode='eval')
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
        img = np.load(os.path.join(self.data_dir, path, 'npy', f'{dir}.npy'))
        img = img[np.newaxis,...] # (y, x, z) -> (1, y, x, z)
        ## load lobe info
        with open(os.path.join(self.data_dir, path, 'npy', 'lobe_info.txt')) as f:
            # 'z_start,z_end\n'
            # 'y_start,y_end,x_start,x_end'
            lobe_info = f.readlines()[-2:]

        Ds, De, Hs, He, Ws, We = crop_with_lobe(*lobe_info, align=align)
        # crop the lobe
        if mode == 'train':
            img = img[:, Hs:He, Ws:We, Ds:De]

        img = np.clip(img, self.clip_min, self.clip_max)
        img = img.astype(np.float32)
        img = img.transpose(0, 3, 1, 2) # (1, y, x, z) -> (1, z, y, x)
        images = img - self.clip_min    # 0 ~ max
        images = (images - self.clip_half) / self.clip_half # -1 ~ 1
        return images, (Ds, De, Hs, He, Ws, We)
    
    def hu_normalize(self, images):
        images = np.array(images, float)
        images = images - self.clip_min
        images = images / (self.clip_max - self.clip_min)
        return images.astype(np.float32)

def pad2factor(image, factor=32, pad_value=170):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image

def fillter_box(bboxes, size):
    res = []
    for box in bboxes:
        if box[0] - box[0+3] / 2 > 0 and box[0] + box[0+3] / 2 < size[0] and \
           box[1] - box[1+3] / 2 > 0 and box[1] + box[1+3] / 2 < size[1] and \
           box[2] - box[2+3] / 2 > 0 and box[2] + box[2+3] / 2 < size[2]:
            res.append(box)
    return np.array(res)

def augment(sample, target, bboxes, do_flip = True, do_rotate=True, do_swap = True):
    #                     angle1 = np.random.rand()*180
    if do_rotate and np.random.randint(0,2):
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if do_swap and np.random.randint(0,2):
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]

    if do_flip and np.random.randint(0,2):
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        # for ax in range(3):
        #     if flipid[ax]==-1:
        #         target[ax] = np.array(sample.shape[ax+1])-target[ax]
        #         bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                if len(bboxes.shape) == 1:
                    print()
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
                # target[ax + 3] = np.array(sample.shape[ax + 1]) - target[ax + 3]
                # bboxes[:, ax + 3] = np.array(sample.shape[ax + 1]) - bboxes[:, ax + 3]
    return sample, target, bboxes

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, lobe=None, isScale=False, isRand=False):
        '''
        img: 3D image loading from npy, (1, d, h, w)
        target: one nodule
        bboxes: all nodules in series
        '''
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        start = []
        for i in range(3):
            # start.append(int(target[i] - crop_size[i] / 2))
            if not isRand:
                # crop the sample base on target
                r = target[i+3]/2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                # crop the sample randomly
                s = np.max([imgs.shape[i+1]-crop_size[i]/2, imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2, imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                i_start = np.random.randint(e, s)
                i_start = max(min(i_start, imgs.shape[i+1]-crop_size[i]),0)
                start.append(i_start)#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))

        coord=[]
        pad = []
        pad.append([0,0])

        for i in range(3):
            leftpad = max(0,-start[i]) # how many pixel need to pad on the left side
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1]) # how many pixel need to pad on the right side
            pad.append([leftpad,rightpad])
            
        crop = imgs[:,
            int(max(start[0],0)):int(min(start[0] + int(crop_size[0]),imgs.shape[1])),
            int(max(start[1],0)):int(min(start[1] + int(crop_size[1]),imgs.shape[2])),
            int(max(start[2],0)):int(min(start[2] + int(crop_size[2]),imgs.shape[3]))]

        crop = np.pad(crop, pad, 'constant', constant_values=0.67142)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=0.67142)
            for i in range(6):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(6):
                    bboxes[i][j] = bboxes[i][j] * scale

        return crop, target, bboxes, coord


import cv2
def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def draw_bbox(image, bbox,filename):
    return
    plt.clf()
    plt.title(filename)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # 不顯示座標軸
    # 解析 bounding box 的坐標
    z,y,x,d ,h,w = bbox 
    x = x - w / 2
    y = y - h / 2
    # 繪製 bounding box
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))  # 綠色邊框
    plt.show()

def draw_all_bbox(image, bbox,filename):
    return 
    plt.clf()
    plt.title(filename)

    z,y,x,d ,h,w = bbox 
    z = int(z - d / 2)
    x = x - w / 2   
    y = y - h / 2
    for i in range(int(d)):
        img = image[0][z+i,:,:]
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # 不顯示座標軸
        
        # 繪製 bounding box
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))  # 綠色邊框
        plt.show()

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