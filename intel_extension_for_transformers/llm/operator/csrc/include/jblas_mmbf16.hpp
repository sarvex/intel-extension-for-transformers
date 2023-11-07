#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include "jblas/jit_blas_wrapper.h"

torch::Tensor jblas_mmbf16_packwei(torch::Tensor& weight, bool transpose);
void jblas_mmbf16(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output);
