PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"
GPU_NUM=4
CONFIG="seg_hrnet_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200_paddle"

$PYTHON -m pip install -r requirements.txt

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/pascal_ctx/$CONFIG.yaml \
        2>&1 | tee local_log.txt
