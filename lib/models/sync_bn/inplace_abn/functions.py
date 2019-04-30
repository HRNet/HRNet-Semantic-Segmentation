from os import path

import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

_src_path = path.join(path.dirname(path.abspath(__file__)), "src")
_backend = load(name="inplace_abn",
                extra_cflags=["-O3"],
                sources=[path.join(_src_path, f) for f in [
                    "inplace_abn.cpp",
                    "inplace_abn_cpu.cpp",
                    "inplace_abn_cuda.cu"
                ]],
                extra_cuda_cflags=["--expt-extended-lambda"])

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            mean, var = _backend.mean_var(x)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            # TODO: implement simplified CUDA backward for inference mode
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = dweight if ctx.affine else None
        dbias = dbias if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                extra, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        cls._parse_extra(ctx, extra)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x) * (ctx.master_queue.maxsize + 1)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            mean, var = _backend.mean_var(x)

            if ctx.is_master:
                means, vars = [mean.unsqueeze(0)], [var.unsqueeze(0)]
                for _ in range(ctx.master_queue.maxsize):
                    mean_w, var_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    means.append(mean_w.unsqueeze(0))
                    vars.append(var_w.unsqueeze(0))

                means = comm.gather(means)
                vars = comm.gather(vars)

                mean = means.mean(0)
                var = (vars + (mean - means) ** 2).mean(0)

                tensors = comm.broadcast_coalesced((mean, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((mean, var))
                mean, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)

            if ctx.is_master:
                edzs, eydzs = [edz], [eydz]
                for _ in range(len(ctx.worker_queues)):
                    edz_w, eydz_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    edzs.append(edz_w)
                    eydzs.append(eydz_w)

                edz = comm.reduce_add(edzs) / (ctx.master_queue.maxsize + 1)
                eydz = comm.reduce_add(eydzs) / (ctx.master_queue.maxsize + 1)

                tensors = comm.broadcast_coalesced((edz, eydz), [edz.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((edz, eydz))
                edz, eydz = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx, dweight, dbias = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = dweight if ctx.affine else None
        dbias = dbias if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]


inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = ["inplace_abn", "inplace_abn_sync", "ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU", "ACT_NONE"]
