# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset

class PASCALContext(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=59,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=520, 
                 crop_size=(480, 480), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],):
    
        super(PASCALContext, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)
        
        self.root = os.path.join(root, 'pascal_ctx/VOCdevkit/VOC2010')
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data
        annots = os.path.join(self.root, 'trainval_merged.json')
        img_path = os.path.join(self.root, 'JPEGImages')
        from detail import Detail
        if 'val' in self.split:
            self.detail = Detail(annots, img_path, 'val')
            mask_file = os.path.join(self.root, 'val.pth')
        elif 'train' in self.split:
            self.mode = 'train'
            self.detail = Detail(annots, img_path, 'train')
            mask_file = os.path.join(self.root, 'train.pth')
        else:
            raise NotImplementedError('only supporting train and val set.')
        self.files = self.detail.getImgs()

        # generate masks
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        
        self._key = np.array(range(len(self._mapping))).astype('uint8')

        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        for i in range(len(self.files)):
            img_id = self.files[i]
            mask = Image.fromarray(self._class_to_index(
                self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        item = self.files[index]
        name = item['file_name']
        img_id = item['image_id']

        image = cv2.imread(os.path.join(self.detail.img_folder,name),
                           cv2.IMREAD_COLOR)
        label = np.asarray(self.masks[img_id],dtype=np.int)
        size = image.shape

        if self.split == 'val':
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            label = cv2.resize(label, self.crop_size, 
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif self.split == 'testval':
            # evaluate model on val dataset
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(label)
        else:
            image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
                                
        return image.copy(), label.copy(), np.array(size), name

    def label_transform(self, label):
        if self.num_classes == 59:
            # background is ignored
            label = np.array(label).astype('int32') - 1
            label[label==-2] = -1
        else:
            label = np.array(label).astype('int32')
        return label
