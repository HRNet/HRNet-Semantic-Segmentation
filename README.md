# High-resolution networks (HRNets) for Semantic Segmentation
## Branches
- This is the implementation for HRNet + OCR.
- The PyTroch 1.1 version ia available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).
- The PyTroch 0.4.1 version is available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master).

## News
- [2020/08/16] [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) has supported our HRNet + OCR.
- [2020/07/20] The researchers from AInnovation have achieved **Rank#1** on [ADE20K Leaderboard](http://sceneparsing.csail.mit.edu/) via training our HRNet + OCR with a semi-supervised learning scheme. More details are in their [Technical Report](https://arxiv.org/pdf/2007.10591.pdf).
- [2020/07/09] Our paper is accepted by ECCV 2020: [Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065.pdf). Notably, the reseachers from Nvidia set a new state-of-the-art performance on Cityscapes leaderboard: [85.4%](https://www.cityscapes-dataset.com/method-details/?submissionID=7836) via combining our HRNet + OCR with a new [hierarchical mult-scale attention scheme](https://arxiv.org/abs/2005.10821). 
- [2020/03/13] Our paper is accepted by TPAMI: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf).
- HRNet + OCR + SegFix: Rank \#1 (84.5) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). OCR: object contextual represenations [pdf](https://arxiv.org/pdf/1909.11065.pdf). ***HRNet + OCR is reproduced [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)***.
- Thanks Google and UIUC researchers. A modified HRNet combined with semantic and instance multi-scale context achieves SOTA panoptic segmentation result on the Mapillary Vista challenge. See [the paper](https://arxiv.org/pdf/1910.04751.pdf).
- Small HRNet models for Cityscapes segmentation. Superior to MobileNetV2Plus ....
- Rank \#1 (83.7) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). HRNet combined with an extension of [object context](https://arxiv.org/pdf/1809.00916.pdf)

- Pytorch-v1.1 and the official Sync-BN supported. We have reproduced the cityscapes results on the new codebase. Please check the [pytorch-v1.1 branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## Introduction
This is the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514). 
We augment the HRNet with a very simple segmentation head shown in the figure below. We aggregate the output representations at four different resolutions, and then use a 1x1 convolutions to fuse these representations. The output representations is fed into the classifier. We evaluate our methods on three datasets, Cityscapes, PASCAL-Context and LIP.

<!-- ![](figures/seg-hrnet.png) -->
<figure>
  <text-align: center;>
  <img src="./figures/seg-hrnet.png" alt="hrnet" title="" width="900" height="150" />
</figcaption>
</figure>

Besides, we further combine HRNet with [Object Contextual Representation](https://arxiv.org/pdf/1909.11065.pdf) and achieve higher performance on the three datasets. The code of HRNet+OCR is contained in this branch. We illustrate the overall framework of OCR in the Figure as shown below:

<figure>
  <text-align: center;>
  <img src="./figures/OCR.PNG" alt="OCR" title="" width="900" height="200" />
</figure>

## Segmentation models
The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification. *Slightly different, we use align_corners = True for upsampling in HRNet*. 

1. Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75.

| model | Train Set | Test Set | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Train | Val | No | No | No | 80.9 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth)/[BaiduYun(Access Code:pmix)](https://pan.baidu.com/s/1KyiOUOR0SYxKtJfIlD5o-w)|
| HRNetV2-W48 + OCR | Train | Val | No | No | No | 81.6 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_8162_torch11.pth)/[BaiduYun(Access Code:fa6i)](https://pan.baidu.com/s/1BGNt4Xmx3yfXUS8yjde0hQ)|
| HRNetV2-W48 + OCR | Train + Val | Test | No | Yes | Yes | 82.3 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_trainval_8227_torch11.pth)/[BaiduYun(Access Code:ycrk)](https://pan.baidu.com/s/16mD81UnGzjUBD-haDQfzIQ)|

2. Performance on the LIP dataset. The models are trained and tested with the input size of 473x473.

| model | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | No | No | Yes | 55.83 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_lip_5583_torch04.pth)/[BaiduYun(Access Code:fahi)](https://pan.baidu.com/s/15DamFiGEoxwDDF1TwuZdnA)|
| HRNetV2-W48 + OCR | No | No | Yes | 56.48 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_lip_5648_torch04.pth)/[BaiduYun(Access Code:xex2)](https://pan.baidu.com/s/1dFYSR2bahRnvpIOdh88kOQ)|

**Note** Currently we could only reproduce HRNet+OCR results on LIP dataset with PyTorch 0.4.1.

3. Performance on the PASCAL-Context dataset. The models are trained and tested with the input size of 520x520.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75,2.0 (the same as EncNet, DANet etc.).

| model |num classes | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | 59 classes | No | Yes | Yes | 54.1 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_pascal_ctx_5410_torch11.pth)/[BaiduYun(Access Code:wz6v)](https://pan.baidu.com/s/1m0MqpHSk0SX380EYEMawSA)|
| HRNetV2-W48 + OCR | 59 classes | No | Yes | Yes | 56.2 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_pascal_ctx_5618_torch11.pth)/[BaiduYun(Access Code:yyxh)](https://pan.baidu.com/s/1XYP54gr3XB76tHmCcKdU9g)|
| HRNetV2-W48 | 60 classes | No | Yes | Yes | 48.3 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gQEHDQrZCiv4R5mf)/[BaiduYun(Access Code:9uf8)](https://pan.baidu.com/s/1pgYt8P8ht2HOOzcA0F7Kag)|
| HRNetV2-W48 + OCR | 60 classes | No | Yes | Yes | 50.1 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_pascal_ctx_5410_torch11.pth)/[BaiduYun(Access Code:gtkb)](https://pan.baidu.com/s/13AYjwzh1LJSlipJwNpJ3Uw)|

4. Performance on the COCO-Stuff dataset. The models are trained and tested with the input size of 520x520.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75,2.0 (the same as EncNet, DANet etc.).

| model | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Yes | No | No | 36.2 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cocostuff_3617_torch04.pth)/[BaiduYun(Access Code:92gw)](https://pan.baidu.com/s/1VAV6KThH1Irzv9HZgLWE2Q)|
| HRNetV2-W48 + OCR | Yes | No | No | 39.7 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cocostuff_3965_torch04.pth)/[BaiduYun(Access Code:sjc4)](https://pan.baidu.com/s/1HFSYyVwKBG3E6y76gcPjDA)|
| HRNetV2-W48 | Yes | Yes | Yes | 37.9 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cocostuff_3617_torch04.pth)/[BaiduYun(Access Code:92gw)](https://pan.baidu.com/s/1VAV6KThH1Irzv9HZgLWE2Q) |
| HRNetV2-W48 + OCR | Yes | Yes | Yes | 40.6 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cocostuff_3965_torch04.pth)/[BaiduYun(Access Code:sjc4)](https://pan.baidu.com/s/1HFSYyVwKBG3E6y76gcPjDA) |

**Note** We reproduce HRNet+OCR results on COCO-Stuff dataset with PyTorch 0.4.1.

5. Performance on the ADE20K dataset. The models are trained and tested with the input size of 520x520.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75,2.0 (the same as EncNet, DANet etc.).

| model | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Yes | No | No | 43.1 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ade20k_4312_torch04.pth)/[BaiduYun(Access Code:f6xf)](https://pan.baidu.com/s/11neVkzxx27qS2-mPFW9dfg)|
| HRNetV2-W48 + OCR | Yes | No | No | 44.5 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_ade20k_4451_torch04.pth)/[BaiduYun(Access Code:peg4)](https://pan.baidu.com/s/1HLhjiLIdgaOHs0SzEtkgkQ)|
| HRNetV2-W48 | Yes | Yes | Yes | 44.2 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ade20k_4312_torch04.pth)/[BaiduYun(Access Code:f6xf)](https://pan.baidu.com/s/11neVkzxx27qS2-mPFW9dfg) |
| HRNetV2-W48 + OCR | Yes | Yes | Yes | 45.5 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_ade20k_4451_torch04.pth)/[BaiduYun(Access Code:peg4)](https://pan.baidu.com/s/1HLhjiLIdgaOHs0SzEtkgkQ) |

**Note** We reproduce HRNet+OCR results on ADE20K dataset with PyTorch 0.4.1.

## Quick start
### Install
1. For LIP dataset, install PyTorch=0.4.1 following the [official instructions](https://pytorch.org/). For Cityscapes and PASCAL-Context, we use PyTorch=1.1.0.
2. `git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT`
3. Install dependencies: pip install -r requirements.txt

If you want to train and evaluate our models on PASCAL-Context, you need to install [details](https://github.com/zhanghang1989/detail-api).
````bash
pip install git+https://github.com/zhanghang1989/detail-api.git#subdirectory=PythonAPI
````

### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/) and [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) datasets.

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── lip
│   ├── TrainVal_images
│   │   ├── train_images
│   │   └── val_images
│   └── TrainVal_parsing_annotations
│       ├── train_segmentations
│       ├── train_segmentations_reversed
│       └── val_segmentations
├── pascal_ctx
│   ├── common
│   ├── PythonAPI
│   ├── res
│   └── VOCdevkit
│       └── VOC2010
├── cocostuff
│   ├── train
│   │   ├── image
│   │   └── label
│   └── val
│       ├── image
│       └── label
├── ade20k
│   ├── train
│   │   ├── image
│   │   └── label
│   └── val
│       ├── image
│       └── label
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
│   ├── lip
│   │   ├── testvalList.txt
│   │   ├── trainList.txt
│   │   └── valList.txt
````

### Train and Test

#### PyTorch Version Differences

Note that the codebase supports both PyTorch 0.4.1 and 1.1.0, and they use different command for training. In the following context, we use `$PY_CMD` to denote different startup command.

```bash
# For PyTorch 0.4.1
PY_CMD="python"
# For PyTorch 1.1.0
PY_CMD="python -m torch.distributed.launch --nproc_per_node=4"
```

e.g., when training on Cityscapes, we use PyTorch 1.1.0. So the command
````bash
$PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````
indicates
````bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````
#### Training

Just specify the configuration file for `tools/train.py`.

For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
$PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````
For example, train the HRNet-W48 + OCR on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
$PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

Note that we only reproduce HRNet+OCR on LIP dataset using PyTorch 0.4.1. So we recommend to use PyTorch 0.4.1 if you want to train on LIP dataset.

#### Testing

For example, evaluating HRNet+OCR on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_ocr_cs_8162_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_ocr_trainval_cs_8227_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the PASCAL-Context validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml \
                     DATASET.TEST_SET testval \
                     TEST.MODEL_FILE hrnet_ocr_pascal_ctx_5618_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the LIP validation set with flip testing:
````bash
python tools/test.py --cfg experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml \
                     DATASET.TEST_SET list/lip/testvalList.txt \
                     TEST.MODEL_FILE hrnet_ocr_lip_5648_torch04.pth \
                     TEST.FLIP_TEST True \
                     TEST.NUM_SAMPLES 0
````
Evaluating HRNet+OCR on the COCO-Stuff validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cocostuff/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr1e-3_wd1e-4_bs_16_epoch110.yaml \
                     DATASET.TEST_SET list/cocostuff/testval.lst \
                     TEST.MODEL_FILE hrnet_ocr_cocostuff_3965_torch04.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.MULTI_SCALE True TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the ADE20K validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/ade20k/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr2e-2_wd1e-4_bs_16_epoch120.yaml \
                     DATASET.TEST_SET list/ade20k/testval.lst \
                     TEST.MODEL_FILE hrnet_ocr_ade20k_4451_torch04.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.MULTI_SCALE True TEST.FLIP_TEST True
````

## Other applications of HRNet
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Image Classification](https://github.com/HRNet/HRNet-Image-Classification)
* [Object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}

@article{YuanCW19,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
  booktitle={ECCV},
  year={2020}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)
    
[2] Object-Contextual Representations for Semantic Segmentation. Yuhui Yuan, Xilin Chen, Jingdong Wang. [download](https://arxiv.org/pdf/1909.11065.pdf)

## Acknowledgement
We adopt sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn) for PyTorch 0.4.1 experiments and the official 
sync-bn provided by PyTorch for PyTorch 1.10 experiments.

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).
