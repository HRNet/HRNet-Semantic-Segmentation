#include <ATen/ATen.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <vector>

#include "common.h"
#include "inplace_abn.h"

// Checks
#ifndef AT_CHECK
  #define AT_CHECK AT_ASSERT
#endif
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utilities
void get_dims(at::Tensor x, int64_t& num, int64_t& chn, int64_t& sp) {
  num = x.size(0);
  chn = x.size(1);
  sp = 1;
  for (int64_t i = 2; i < x.ndimension(); ++i)
    sp *= x.size(i);
}

// Operations for reduce
template<typename T>
struct SumOp {
  __device__ SumOp(const T *t, int c, int s)
      : tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ T operator()(int batch, int plane, int n) {
    return tensor[(batch * chn + plane) * sp + n];
  }
  const T *tensor;
  const int chn;
  const int sp;
};

template<typename T>
struct VarOp {
  __device__ VarOp(T m, const T *t, int c, int s)
      : mean(m), tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ T operator()(int batch, int plane, int n) {
    T val = tensor[(batch * chn + plane) * sp + n];
    return (val - mean) * (val - mean);
  }
  const T mean;
  const T *tensor;
  const int chn;
  const int sp;
};

template<typename T>
struct GradOp {
  __device__ GradOp(T _weight, T _bias, const T *_z, const T *_dz, int c, int s)
      : weight(_weight), bias(_bias), z(_z), dz(_dz), chn(c), sp(s) {}
  __device__ __forceinline__ Pair<T> operator()(int batch, int plane, int n) {
    T _y = (z[(batch * chn + plane) * sp + n] - bias) / weight;
    T _dz = dz[(batch * chn + plane) * sp + n];
    return Pair<T>(_dz, _y * _dz);
  }
  const T weight;
  const T bias;
  const T *z;
  const T *dz;
  const int chn;
  const int sp;
};

/***********
 * mean_var
 ***********/

template<typename T>
__global__ void mean_var_kernel(const T *x, T *mean, T *var, int num, int chn, int sp) {
  int plane = blockIdx.x;
  T norm = T(1) / T(num * sp);

  T _mean = reduce<T, SumOp<T>>(SumOp<T>(x, chn, sp), plane, num, chn, sp) * norm;
  __syncthreads();
  T _var = reduce<T, VarOp<T>>(VarOp<T>(_mean, x, chn, sp), plane, num, chn, sp) * norm;

  if (threadIdx.x == 0) {
    mean[plane] = _mean;
    var[plane] = _var;
  }
}

std::vector<at::Tensor> mean_var_cuda(at::Tensor x) {
  CHECK_INPUT(x);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto mean = at::empty(x.type(), {chn});
  auto var = at::empty(x.type(), {chn});

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(x.type(), "mean_var_cuda", ([&] {
    mean_var_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        num, chn, sp);
  }));

  return {mean, var};
}

/**********
 * forward
 **********/

template<typename T>
__global__ void forward_kernel(T *x, const T *mean, const T *var, const T *weight, const T *bias,
                               bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  T _mean = mean[plane];
  T _var = var[plane];
  T _weight = affine ? abs(weight[plane]) + eps : T(1);
  T _bias = affine ? bias[plane] : T(0);

  T mul = rsqrt(_var + eps) * _weight;

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      T _x = x[(batch * chn + plane) * sp + n];
      T _y = (_x - _mean) * mul + _bias;

      x[(batch * chn + plane) * sp + n] = _y;
    }
  }
}

at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(mean);
  CHECK_INPUT(var);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(x.type(), "forward_cuda", ([&] {
    forward_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        affine, eps, num, chn, sp);
  }));

  return x;
}

/***********
 * edz_eydz
 ***********/

template<typename T>
__global__ void edz_eydz_kernel(const T *z, const T *dz, const T *weight, const T *bias,
                                T *edz, T *eydz, bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  T _weight = affine ? abs(weight[plane]) + eps : 1.f;
  T _bias = affine ? bias[plane] : 0.f;

  Pair<T> res = reduce<Pair<T>, GradOp<T>>(GradOp<T>(_weight, _bias, z, dz, chn, sp), plane, num, chn, sp);
  __syncthreads();

  if (threadIdx.x == 0) {
    edz[plane] = res.v1;
    eydz[plane] = res.v2;
  }
}

std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                      bool affine, float eps) {
  CHECK_INPUT(z);
  CHECK_INPUT(dz);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto edz = at::empty(z.type(), {chn});
  auto eydz = at::empty(z.type(), {chn});

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(z.type(), "edz_eydz_cuda", ([&] {
    edz_eydz_kernel<scalar_t><<<blocks, threads>>>(
        z.data<scalar_t>(),
        dz.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        edz.data<scalar_t>(),
        eydz.data<scalar_t>(),
        affine, eps, num, chn, sp);
  }));

  return {edz, eydz};
}

/***********
 * backward
 ***********/

template<typename T>
__global__ void backward_kernel(const T *z, const T *dz, const T *var, const T *weight, const T *bias, const T *edz,
                                const T *eydz, T *dx, T *dweight, T *dbias,
                                bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  T _weight = affine ? abs(weight[plane]) + eps : 1.f;
  T _bias = affine ? bias[plane] : 0.f;
  T _var = var[plane];
  T _edz = edz[plane];
  T _eydz = eydz[plane];

  T _mul = _weight * rsqrt(_var + eps);
  T count = T(num * sp);

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      T _dz = dz[(batch * chn + plane) * sp + n];
      T _y = (z[(batch * chn + plane) * sp + n] - _bias) / _weight;

      dx[(batch * chn + plane) * sp + n] = (_dz - _edz / count - _y * _eydz / count) * _mul;
    }
  }

  if (threadIdx.x == 0) {
    if (affine) {
      dweight[plane] = weight[plane] > 0 ? _eydz : -_eydz;
      dbias[plane] = _edz;
    }
  }
}

std::vector<at::Tensor> backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                      at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  CHECK_INPUT(z);
  CHECK_INPUT(dz);
  CHECK_INPUT(var);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  CHECK_INPUT(edz);
  CHECK_INPUT(eydz);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto dx = at::zeros_like(z);
  auto dweight = at::zeros_like(weight);
  auto dbias = at::zeros_like(bias);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(z.type(), "backward_cuda", ([&] {
    backward_kernel<scalar_t><<<blocks, threads>>>(
        z.data<scalar_t>(),
        dz.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        edz.data<scalar_t>(),
        eydz.data<scalar_t>(),
        dx.data<scalar_t>(),
        dweight.data<scalar_t>(),
        dbias.data<scalar_t>(),
        affine, eps, num, chn, sp);
  }));

  return {dx, dweight, dbias};
}

/**************
 * activations
 **************/

template<typename T>
inline void leaky_relu_backward_impl(T *z, T *dz, float slope, int64_t count) {
  // Create thrust pointers
  thrust::device_ptr<T> th_z = thrust::device_pointer_cast(z);
  thrust::device_ptr<T> th_dz = thrust::device_pointer_cast(dz);

  thrust::transform_if(th_dz, th_dz + count, th_z, th_dz,
                       [slope] __device__ (const T& dz) { return dz * slope; },
                       [] __device__ (const T& z) { return z < 0; });
  thrust::transform_if(th_z, th_z + count, th_z,
                       [slope] __device__ (const T& z) { return z / slope; },
                       [] __device__ (const T& z) { return z < 0; });
}

void leaky_relu_backward_cuda(at::Tensor z, at::Tensor dz, float slope) {
  CHECK_INPUT(z);
  CHECK_INPUT(dz);

  int64_t count = z.numel();

  AT_DISPATCH_FLOATING_TYPES(z.type(), "leaky_relu_backward_cuda", ([&] {
    leaky_relu_backward_impl<scalar_t>(z.data<scalar_t>(), dz.data<scalar_t>(), slope, count);
  }));
}

template<typename T>
inline void elu_backward_impl(T *z, T *dz, int64_t count) {
  // Create thrust pointers
  thrust::device_ptr<T> th_z = thrust::device_pointer_cast(z);
  thrust::device_ptr<T> th_dz = thrust::device_pointer_cast(dz);

  thrust::transform_if(th_dz, th_dz + count, th_z, th_z, th_dz,
                       [] __device__ (const T& dz, const T& z) { return dz * (z + 1.); },
                       [] __device__ (const T& z) { return z < 0; });
  thrust::transform_if(th_z, th_z + count, th_z,
                       [] __device__ (const T& z) { return log1p(z); },
                       [] __device__ (const T& z) { return z < 0; });
}

void elu_backward_cuda(at::Tensor z, at::Tensor dz) {
  CHECK_INPUT(z);
  CHECK_INPUT(dz);

  int64_t count = z.numel();

  AT_DISPATCH_FLOATING_TYPES(z.type(), "leaky_relu_backward_cuda", ([&] {
    elu_backward_impl<scalar_t>(z.data<scalar_t>(), dz.data<scalar_t>(), count);
  }));
}
