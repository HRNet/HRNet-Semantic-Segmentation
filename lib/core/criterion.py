# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()
        
        
        
        
class CrossEntropy_AUX(nn.Module):
    def __init__(self, ignore_label=-1, weight=None,model_name=None):
        super(CrossEntropy_AUX, self).__init__()
        self.ignore_label = ignore_label
        self.criterion0 = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
        self.criterion1 = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
        self.model_name=model_name

    def forward(self, score, target):
        score_aux = score[0]
        score_seg = score[1]
        ph0, pw0 = score_aux.size(2), score_aux.size(3)
        ph1, pw1 = score_seg.size(2), score_seg.size(3)
        
        h, w = target.size(1), target.size(2)
        if ph0 != h or pw0 != w:
            if "alignTrue" in self.model_name:
                score_aux = F.upsample(
                    input=score_aux, size=(h, w), mode='bilinear', align_corners=True)
            else:
                score_aux = F.upsample(
                    input=score_aux, size=(h, w), mode='bilinear', align_corners=False)
        if ph1 != h or pw1 != w:
            if "alignTrue" in self.model_name:
                score_seg = F.upsample(
                    input=score_seg, size=(h, w), mode='bilinear', align_corners=True)
            else:
                score_seg = F.upsample(
                    input=score_seg, size=(h, w), mode='bilinear', align_corners=False)

        loss = 0.4 * self.criterion0(score_aux, target) + 1.0 * self.criterion1(score_seg, target)

        return loss