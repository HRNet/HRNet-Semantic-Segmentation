#!/bin/bash

sudo nvidia-docker run --ipc=host -t -v $HOME/HRNet_OCR-0.4:/workspace/hrnetocr -v /mnt/openseg/data:/mnt/openseg/data  kesun/pose:pytorch-0.4.1-bn-cudnn-off bash -c "cd hrnetocr; pip install -r requirements.txt; python tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_bndefault_alignTrue_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_1x1.yaml"