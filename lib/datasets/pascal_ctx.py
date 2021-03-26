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

import torch
from torch.nn import functional as F
from PIL import Image

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
                 std=[0.229, 0.224, 0.225]):

        super(PASCALContext, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )
        if self.num_classes == 59:
            label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image, border_padding = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name, border_padding

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name