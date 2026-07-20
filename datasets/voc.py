import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import imageio
from . import transforms
import torchvision
from PIL import Image
from torchvision import transforms as T
from utils import imutils

import random

class_list = ["_background_", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str) 
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item() 

class Normalize:
    def __call__(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        return image

class VOC12Dataset(Dataset): 
    def __init__( 
        self,
        root_dir=None, 
        name_list_dir=None, 
        split='train', 
        stage='train', 
    ):
        super().__init__()

        self.root_dir = root_dir 
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages') 
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')  
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label 

class VOC12ClsDataset(VOC12Dataset): 
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],  
                 rescale_range=[0.5, 2.0],  
                 img_fliplr=True, 
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 数据增强：基础的颜色抖动和翻转
        self.basic_augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            ], p=0.5),
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        if self.aug:
            # 应用基础的几何变换
            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            
           
            image = np.clip(image, 0, 255).astype(np.uint8)
            
           
            pil_image = Image.fromarray(image)
            image = self.basic_augment(pil_image)
        else:
          
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        
     
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
      
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        
        return image 

    
    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

       
        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1 
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)

        image = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        return img_name, image, cls_label 


class VOC12SegDataset(VOC12Dataset): 
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 rescale=None,
                 crop_size=512,
                 crop_method="interpolate",
                 hor_flip=False,
                 img_normal=Normalize(),
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.img_normal = img_normal
        self.rescale = rescale
        self.crop_size = crop_size
        self.crop_method = crop_method
        self.hor_flip = hor_flip

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        
        if self.rescale:
            image, label = imutils.random_scale((image, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            image = self.img_normal(image)

        if self.hor_flip:
            image, label = imutils.random_lr_flip((image, label))

        if self.crop_method == "random":
            image, label = imutils.random_crop((image, label), self.crop_size, (0, 255))
        elif self.crop_method == "interpolate":
            image = cv2.resize(image, dsize=(self.crop_size, self.crop_size))
            # label = cv2.resize(label, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        else:
            image = imutils.top_left_crop(image, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        image = imutils.HWC_to_CHW(image)

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)
        
        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label
