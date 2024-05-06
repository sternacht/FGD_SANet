import math
import numpy as np
import warnings
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate


def pad2factor(image, factor=32, pad_value=170):
    depth, height, width = image.shape[-3:]
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

class Augment(object):
    def __init__(self, config):
        self.do_rotate = config['augtype']['rotate']
        self.do_swap = config['augtype']['swap']
        self.do_flip = config['augtype']['flip']

    def __call__(self, sample, target, bboxes):
        if self.do_rotate and np.random.randint(0,2):
            sample, target, bboxes = self.img_rotate(sample, target, bboxes)
            
        if self.do_swap and np.random.randint(0,2):
            sample, target, bboxes = self.img_swap(sample, target, bboxes)

        if self.do_flip and np.random.randint(0,2):
            sample, target, bboxes = self.img_flip(sample, target, bboxes)

        return sample, target, bboxes

    def img_rotate(self, sample, target, bboxes):
        #                     angle1 = np.random.rand()*180
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
        return sample, target, bboxes

    def img_swap(self, sample, target, bboxes):
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            # at least swap on two of any dimension
            while True:
                axisorder = np.random.permutation(3)
                if not np.all(axisorder, np.array([0,1,2])):
                    break
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
        return sample, target, bboxes

    def img_flip(self, sample, target, bboxes, flipid=None):
        # at least flip in one of any dimensions
        if isinstance(flipid, list) or isinstance(flipid, np.ndarray):
            if len(flipid) != 3:
                warnings.warn("invalid flipid len or shape")
        else:
            while True:
                flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
                if flipid.sum() < 3:
                    break
        # flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
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

