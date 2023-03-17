import random
import numpy as np
import cv2
import torch

from patchify import patchify, unpatchify

import albumentations
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import imgaug as ia
from imgaug import augmenters as iaa
import imageio

from .edge_utils import mask_to_onehot, onehot_to_multiclass_edges, onehot_to_binary_edges
def extract_ordered_patches(imgs, patch_size:tuple,stride_size:tuple):#(imgs,(100,100),(100,100))
        assert imgs.ndim > 2
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, axis=0)
        
        b,h,w,c = imgs.shape
        patch_h,patch_w = patch_size
        stride_h,stride_w = stride_size
        assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
        #y方向的切片数量
        n_patches_y = (h - patch_h) // stride_h + 1
        #x方向的切片数量
        n_patches_x = (w - patch_w) // stride_w + 1
        #每张图片的切片数量
        n_patches_per_img = n_patches_y*n_patches_x
        #切片总数
        n_patchs = n_patches_per_img*b
        #设置图像块大小
        # patches = torch.from_numpy(np.empty((n_patchs,patch_h,patch_w,c),dtype = imgs.dtype))
        patches = np.empty((n_patchs, patch_h, patch_w,c ),dtype = imgs.dtype)
        #每张图切出相同数量相同大小的切片，计算出各个切片的位置，从图中取出对应的部分就得到各切片。
        #依次对每张图处理
        patch_idx = 0
        for img in imgs:
            #从上到下、从左到右依次切片
            for i in range(n_patches_y):
                for j in range(n_patches_x):
                    y1 = i*stride_h
                    y2 = y1+patch_h
                    x1 = j*stride_w
                    x2 = x1+patch_w
                    patches[patch_idx] = img[y1:y2,x1:x2]
                    patch_idx += 1
        return patches


class BaseDataSet(Dataset):
    def __init__(self, num_classes, root, split, mean, std, base_size=None, augment=True, val=False,
                crop=True, rotate90=True, flip=False, shift_scale_rotate=True, blur=False, 
                enhancement=False, return_id=True, return_edge=False,transpose=False,
                gamma=False,grid=False,Elastic=False,hue=False,clahe=False,contract=False):
        self.num_classes = num_classes
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop = crop
        self.base_size = base_size
        if self.augment:
            self.rotate90 = rotate90
            self.flip = flip
            self.shift_scale_rotate = shift_scale_rotate
            self.blur = blur
            self.contract = contract
            self.enhancement = enhancement
            self.transpose = transpose
            self.gamma = gamma
            self.grid  = grid
            self.Elastic = Elastic
            self.hue = hue
            self.clahe = clahe
            
            
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id
        self.return_edge = return_edge

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError
    
    def _augmentation(self, image: np.ndarray, mask: np.ndarray):
        # 几何变换
        augmentations_geometry = []
        # 色彩变换
        augmentations_color = []
        H, W = self.base_size
        augmentations_geometry.append(albumentations.Resize(W, H))

        if self.flip:
            augmentations_geometry.append(albumentations.Flip())
        
        if self.rotate90:
            augmentations_geometry.append(albumentations.RandomRotate90(always_apply=False, p=0.5)) # 长宽比不一致旋转会改变长宽比
        
        if self.shift_scale_rotate:
            # augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.4))#大豆
            # augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.4))#大豆16.06
            #augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, p=1))#大豆18.02
            augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.2))#猪肉
            # augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=1))
            # augmentations_geometry.append(albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=0.5))
            
        
        if self.crop:
            augmentations_geometry.append(albumentations.RandomResizedCrop(H, W, scale=(0.8, 1)))
    
        
        if self.enhancement:
            # 直方图增强、锐化
            # augmentations_color.append(albumentations.OneOf([
            #     albumentations.CLAHE(clip_limit=2),
            #     albumentations.IAASharpen()
            # ], p=0.3))
            augmentations_color.append(
                albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(3, 3), always_apply=False, p=0.3))

        # if self.clahe:
            
        #     augmentations_color2.append(
        #         iaa.Sometimes(0.3, [
        #                             iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1.5, 4.0))
        #                         ]))

        if self.contract:
            augmentations_color.append(
                albumentations.RandomContrast(limit=(0.5,1.5), p=0.4))

        if self.blur:
            # augmentations_color.append(albumentations.OneOf([
            #     albumentations.MotionBlur(p=0.2),
            #     albumentations.MedianBlur(blur_limit=3, p=0.2),
            #     albumentations.Blur(blur_limit=5, p=0.2)
            # ], p=0.2))
            augmentations_color.append(albumentations.Blur(blur_limit=5, p=0.1))
            
        if self.transpose:
            augmentations_geometry.append(albumentations.Transpose(always_apply=False, p=0.5))

        if self.gamma:
            augmentations_color.append(albumentations.RandomGamma(gamma_limit=(40, 120), eps=1e-07, always_apply=False, p=0.2))

        if self.grid:
            augmentations_geometry.append(albumentations.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.3))

        if self.Elastic:
            augmentations_geometry.append(albumentations.ElasticTransform(alpha = 1,sigma = 50,alpha_affine = 50,interpolation = 1,border_mode = 4,value = None,mask_value = None,always_apply = False,approximate = False,p = 0.3 ))

        if self.hue:
            augmentations_color.append(albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.1))

        
        

        augmentations_geometry.append(albumentations.Resize(H, W)) # 全部resize至一个尺寸

        transforms_geometry = albumentations.Compose(augmentations_geometry)
        transforms_color = albumentations.Compose(augmentations_color)


        augmented_geo = transforms_geometry(image=image, mask=mask)
        image, mask = augmented_geo['image'], augmented_geo['mask']
        

        image = transforms_color(image=image)['image']
        
        return image, mask

    def _val_augmentation(self, image, mask):
        H, W = self.base_size
        auged = albumentations.Resize(W, H)(image=image, mask=mask)
        image, mask = auged['image'], auged['mask']
        return image, mask
        # if self.crop_size:
        #     h, w = label.shape
        #     # # Scale the smaller side to crop size
        #     # if h < w:
        #     #     h, w = (self.crop_size, int(self.crop_size * w / h))
        #     # else:
        #     #     h, w = (int(self.crop_size * h / w), self.crop_size)
        #     longside = self.crop_size
        #     h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)

        #     image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        #     label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        #     label = np.asarray(label, dtype=np.int32)

        #     # # Center Crop
        #     # h, w = label.shape
        #     # start_h = (h - self.crop_size )// 2
        #     # start_w = (w - self.crop_size )// 2
        #     # end_h = start_h + self.crop_size
        #     # end_w = start_w + self.crop_size
        #     # image = image[start_h:end_h, start_w:end_w]
        #     # label = label[start_h:end_h, start_w:end_w]
        # return image, label

    # def _augmentation(self, image, label):
    #     h, w, _ = image.shape
    #     # Scaling, we set the bigger to base size, and the smaller 
    #     # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing

    #     if self.base_size:
    #         if self.scale:
    #             # 缩放比例
    #             longside = random.randint(int(self.base_size*0.8), int(self.base_size*1.5))
    #         else:
    #             longside = self.base_size
    #         h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
    #         image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    #         label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
    #     h, w, _ = image.shape
    #     # Rotate the image with an angle between -10 and 10
    #     if self.rotate:
    #         angle = random.randint(-10, 10)
    #         center = (w / 2, h / 2)
    #         rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    #         image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
    #         label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

    #     # Padding to return the correct crop size
    #     if self.crop_size:
    #         pad_h = max(self.crop_size - h, 0)
    #         pad_w = max(self.crop_size - w, 0)
    #         pad_kwargs = {
    #             "top": 0,
    #             "bottom": pad_h,
    #             "left": 0,
    #             "right": pad_w,
    #             "borderType": cv2.BORDER_CONSTANT,}
    #         if pad_h > 0 or pad_w > 0:
    #             image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
    #             label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
    #         # Cropping 
    #         h, w, _ = image.shape
    #         start_h = random.randint(0, h - self.crop_size)
    #         start_w = random.randint(0, w - self.crop_size)
    #         end_h = start_h + self.crop_size
    #         end_w = start_w + self.crop_size
    #         image = image[start_h:end_h, start_w:end_w]
    #         label = label[start_h:end_h, start_w:end_w]

    #     # Random H flip
    #     if self.flip:
    #         if random.random() > 0.5:
    #             image = np.fliplr(image).copy()
    #             label = np.fliplr(label).copy()

    #     # Gaussian Blud (sigma between 0 and 1.5)
    #     if self.blur:
    #         sigma = random.random()
    #         ksize = int(3.3 * sigma)
    #         ksize = ksize + 1 if ksize % 2 == 0 else ksize
    #         image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    #     return image, label
       
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        #修改的地方，修改image类型为uint8
        image = image.astype(np.uint8)
        
        #修改结束
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        # patches = extract_ordered_patches(image,(200,200),(200,200))#8,3,200,200
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        # patches2 = torch.empty(len(patches),3,200,200)

        # for i in range(len(patches)):
        #     image = Image.fromarray(np.uint8(patches[i,:,:,:]))
        #     image = self.normalize(self.to_tensor(image))
        #     patches2[i,:,:,:] = image

        image = Image.fromarray(np.uint8(image))
        #修改的地方
        image = self.normalize(self.to_tensor(image))
        # image = self.to_tensor(image)/255
        #修改结束
        
        return_dict = {
            'image': image, 
            'label': label
            }
        if self.return_id:
            return_dict['image_id'] = image_id
        if self.return_edge:
            edge_one_hot = mask_to_onehot(label, self.num_classes) # !!从1开始数
            edge_binary = onehot_to_binary_edges(edge_one_hot, 2, self.num_classes) # 半径为2的边界
            return_dict['edge_binary'] = edge_binary
        return return_dict

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

