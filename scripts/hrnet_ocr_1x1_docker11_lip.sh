#!/bin/bash

sudo nvidia-docker run --ipc=host -t -v $HOME/HRNet_OCR-0.4:/workspace/hrnetocr -v /mnt:/mnt -v /msravcshare:/msravcshare chaoqw/hrnet_segmentation:pytorchv1.1-cu10_pascal bash -c 'cd /workspace/hrnetocr/; NCCL_LL_THRESHOLD=0 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/lip/seg_hrnet_ocr_bndefault_alignTrue_1x1_w48_473x473_sgd_lr7e-3_wd5e-4_bs_32_epoch150.yaml MODEL.PRETRAINED pretrained_models/hrnetv2_w48_imagenet_pretrained.pth'
