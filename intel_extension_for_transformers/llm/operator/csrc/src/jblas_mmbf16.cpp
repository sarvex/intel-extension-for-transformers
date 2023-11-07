#include "../include/jblas_mmbf16.hpp"
#include "jblas/jit_blas_prologue.h"
#include "jblas/jit_blas_utils.h"

torch::Tensor jblas_mmbf16_packwei(torch::Tensor& weight, bool transpose) {
  using packKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
  TORCH_CHECK(weight.sizes() == 2, "Qbits: only support 2-dim weight in mmbf16_packwei.");
  static packKernel kernel;
  auto n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  auto k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  auto packw = kernel.getWeightPtr()->createStorage(n, k);
  auto pack_wei_tensor = torch::zeros(packw.mSize, torch::kInt8);
  packw.assign(pack_wei_tensor.data_ptr<int8_t>());
  if (transpose) {
    kernel.getWeightPtr()->packWeightTranspose(n, k,
                                               {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), n, &packw});
  } else {
    kernel.getWeightPtr()->packWeight(n, k, {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), k, &packw});
  }
  return pack_wei_tensor;
}

void jblas_mmbf16(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output) {
  using GEMMKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
  static GEMMKernel kernel;
  auto deseries_wei = jblas::prologue::gemm::PackedWeightParser::deserialBuffer(weight.data_ptr());
  int m = output.sizes()[0];
  int n = output.sizes()[1];
  int k = activation.sizes()[1];
  GEMMKernel::Arguments args{m,
                             n,
                             k,
                             reinterpret_cast<jblas::utils::bf16*>(activation.data_ptr()),
                             k,
                             NULL,
                             0,
                             reinterpret_cast<jblas::prologue::gemm::StoragePackedWeight*>(deseries_wei),
                             reinterpret_cast<jblas::utils::bf16*>(output.data_ptr()),
                             n,
                             NULL};
  kernel.compute(args);
}