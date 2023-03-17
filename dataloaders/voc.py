# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, num_classes=21, ignore_indexes=[], **kwargs):
        self.num_classes = num_classes # 分割类别个数
        self.index2class_name = {} # 索引及对应的类别名称
        self.raw_num_classes = num_classes + len(ignore_indexes) # 原始分类标签个数
        self.ignore_indexes = sorted(ignore_indexes, reverse=False) # 忽略类别列表
        self.palette = palette.get_voc_palette(self.raw_num_classes, ignore_indexes) # 使用原始类别个数绘制调色板，保证颜色映射不变
        super(VOCDataset, self).__init__(num_classes=num_classes, **kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, '')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, "ImageSets", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        # image = cv2.imread(image_path)
        # image = cv2.pyrMeanShiftFiltering(image, sp=5, sr=10)#5,10
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.asarray(image, dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # 忽略指定类别
        reset_classes = list(range(self.raw_num_classes))
        # 忽略类别置0
        for idx in self.ignore_indexes:
            reset_classes[idx] = 0
            label[label == idx] = 0
        # 原始类别重新映射索引
        reset_classes = list(set(reset_classes))
        for i, v in enumerate(reset_classes):
            label[label == v] = i
        
        return image, label, image_id

class VOCAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, num_classes=21, ignore_indexes=[], **kwargs):
        self.num_classes = num_classes
        self.ignore_indexes = sorted(ignore_indexes, reverse=False)
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, '')

        file_list = os.path.join(self.root, "ImageSets", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))
    
    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, num_classes, ignore_indexes, data_dir, batch_size, split, base_size=None, num_workers=1, val=False,
                    shuffle=False, augment=False, crop=True, rotate90=True, flip=True, shift_scale_rotate=False,
                     blur=False, enhancement=False, drop_last=False, val_split= None, return_id=True,
                     transpose=False,gamma=False,grid=False,Elastic=False,hue=False,clahe=False,contract=False):
        
        # self.MEAN = [0.03214175, 0.043852817, 0.01769489]
        # self.STD = [0.06753985, 0.090656504, 0.03844671]
        # self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        # self.STD = [0.23965294, 0.23532275, 0.2398498]
        #ImageNet上的
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop': crop,
            'base_size': base_size,
            'flip': flip,
            'blur': blur,
            'rotate90': rotate90,
            'shift_scale_rotate': shift_scale_rotate,
            'enhancement': enhancement,
            'return_id': return_id,
            'val': val,
            'contract': contract,
            "gamma":gamma,
            "transpose":transpose,
            "grid":grid,
            "Elastic":Elastic,
            "hue":hue,
            "clahe":clahe
        }
    
        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(num_classes=num_classes, ignore_indexes=ignore_indexes, **kwargs)
        elif split in ["train", "trainval", "val", "test", 'train_0', 'val_0', 'train_DUT', 'val_DUT','train_1', 'val_1','train_2', 'val_2','train_3', 'val_3','train_4', 'val_4','val_test','test_2','train_2_']:
            self.dataset = VOCDataset(num_classes=num_classes, ignore_indexes=ignore_indexes, **kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(VOC, self).__init__(self.dataset, batch_size, shuffle, num_workers, drop_last, val_split)

