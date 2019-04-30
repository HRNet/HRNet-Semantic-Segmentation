#pragma once

#include <ATen/ATen.h>

#include <vector>

std::vector<at::Tensor> mean_var_cpu(at::Tensor x);
std::vector<at::Tensor> mean_var_cuda(at::Tensor x);

at::Tensor forward_cpu(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                       bool affine, float eps);
at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps);

std::vector<at::Tensor> edz_eydz_cpu(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                     bool affine, float eps);
std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                      bool affine, float eps);

std::vector<at::Tensor> backward_cpu(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                     at::Tensor edz, at::Tensor eydz, bool affine, float eps);
std::vector<at::Tensor> backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                      at::Tensor edz, at::Tensor eydz, bool affine, float eps);

void leaky_relu_backward_cpu(at::Tensor z, at::Tensor dz, float slope);
void leaky_relu_backward_cuda(at::Tensor z, at::Tensor dz, float slope);

void elu_backward_cpu(at::Tensor z, at::Tensor dz);
void elu_backward_cuda(at::Tensor z, at::Tensor dz);