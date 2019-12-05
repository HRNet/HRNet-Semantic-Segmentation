import torch
import torch.distributed as torch_dist

def is_distributed():
    return torch_dist.is_initialized()

def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()

def get_rank():
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()

# def get_device():
#     if is_distributed():
#         return torch.device('cuda:{}'.format(get_rank()))
#     else:
#         return torch.cuda.current_device()