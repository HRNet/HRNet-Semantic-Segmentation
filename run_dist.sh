PYTHON="/opt/conda/bin/python"
GPU_NUM=$1
CONFIG=$2

$PYTHON -m pip install -r requirements.txt

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/$CONFIG.yaml \
        2>&1 | tee local_log.txt
