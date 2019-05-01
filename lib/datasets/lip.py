# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
 
class LIP(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=20,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=473, 
                 crop_size=(473, 473), 
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(LIP, self).__init__(ignore_label, base_size,
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
            if 'train' in self.list_path:
                image_path, label_path, label_rev_path, _ = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "label_rev": label_rev_path, 
                          "name": name,}
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name,}
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, label, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
     
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
         
        image = cv2.imread(os.path.join(
                    self.root, 'lip/TrainVal_images/', item["img"]), 
                    cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(
                    self.root, 'lip/TrainVal_parsing_annotations/', 
                    item["label"]),
                    cv2.IMREAD_GRAYSCALE)
        size = label.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :] 
            if flip == -1:
                label = cv2.imread(os.path.join(self.root, 
                            'lip/TrainVal_parsing_annotations/', 
                            item["label_rev"]),
                            cv2.IMREAD_GRAYSCALE)
        
        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                          size=(size[-2], size[-1]), 
                          mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:,14,:,:] = flip_output[:,15,:,:]
            flip_pred[:,15,:,:] = flip_output[:,14,:,:]
            flip_pred[:,16,:,:] = flip_output[:,17,:,:]
            flip_pred[:,17,:,:] = flip_output[:,16,:,:]
            flip_pred[:,18,:,:] = flip_output[:,19,:,:]
            flip_pred[:,19,:,:] = flip_output[:,18,:,:]
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
