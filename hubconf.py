"""File for accessing HRNet via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('AlexeyAB/PyTorch_YOLOv4:u5_preview', 'yolov4_pacsp_s', pretrained=True, channels=3, classes=80)
"""

dependencies = ['torch']
import torch
from lib.models.seg_hrnet import get_seg_model


state_dict_url = 'https://github.com/huawei-noah/ghostnet/raw/master/pytorch/models/state_dict_93.98.pth'


def hrnet_w48_cityscapes(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    HRNetW48 model pretrained on Cityscapes
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model