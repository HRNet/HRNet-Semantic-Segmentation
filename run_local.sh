PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"
GPU_NUM=4
CONFIG="seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484"

$PYTHON -m pip install -r requirements.txt

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/cityscapes/$CONFIG.yaml \
        2>&1 | tee local_log.txt
