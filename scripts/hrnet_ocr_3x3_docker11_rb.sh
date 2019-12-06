#!/bin/bash

sudo nvidia-docker run --ipc=host -t -v $HOME/HRNet_OCR-0.4:/workspace/hrnetocr -v /mnt/openseg/data:/mnt/openseg/data -v /msravcshare:/msravcshare chaoqw/hrnet_segmentation:pytorchv1.1-cu10_pascal bash -c 'cd /workspace/hrnetocr/; random_brightness=1 NCCL_LL_THRESHOLD=0 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_bndefault_alignTrue_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_3x3.yaml MODEL.PRETRAINED pretrained_models/hrnetv2_w48_imagenet_pretrained.pth'
