//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include "../include/jblas_mmbf16.hpp"
#include "jblas/jit_blas_prologue.h"
#include "jblas/jit_blas_utils.h"
#include "jblas/jit_blas_wrapper.h"
#include "../dispatcher/include/jblas_weightonly_dispatcher.hpp"

torch::Tensor jblas_mmbf16_packwei(torch::Tensor& weight, bool transpose) {
  if (initer.verbose) timer.start();
  using packKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
  TORCH_CHECK(weight.dim() == 2, "Qbits: only support 2-dim weight in mmbf16_packwei.");
  static packKernel kernel;
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  auto packw = kernel.getWeightPtr()->createStorage(n, k);
  auto pack_wei_tensor = torch::empty(packw.mSize, torch::kInt8);
  packw.assign(pack_wei_tensor.data_ptr<int8_t>());
  if (transpose) {
    kernel.getWeightPtr()->packWeightTranspose(n, k,
                                               {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), k, &packw});
  } else {
    kernel.getWeightPtr()->packWeight(n, k, {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), n, &packw});
  }
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits mmbf16-packwei verbose\nn:" << n << " k:" << k << " execute time:" << cost_time << "ms";
  }
  return pack_wei_tensor;
}

torch::Tensor jblas_mmfp32_avx2_packwei(torch::Tensor& weight, bool transpose) {
  if (initer.verbose) timer.start();
  using packKernel = jblas::wrapper::gemm_default::avx2::GemmKernel;
  static packKernel kernel;
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  auto packw = kernel.getWeightPtr()->createStorage(n, k);
  auto pack_wei_tensor = torch::empty(packw.mSize, torch::kInt8);
  packw.assign(pack_wei_tensor.data_ptr<int8_t>());
  if (transpose) {
    kernel.getWeightPtr()->packWeightTranspose(n, k, {reinterpret_cast<float*>(weight.data_ptr()), k, &packw});
  } else {
    kernel.getWeightPtr()->packWeight(n, k, {reinterpret_cast<float*>(weight.data_ptr()), n, &packw});
  }
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits mmfp32-avx2-packwei verbose\nn:" << n << " k:" << k << " execute time:" << cost_time << "ms";
  }
  return pack_wei_tensor;
}

void jblas_mmbf16(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output) {
  if (initer.verbose) timer.start();
  using GEMMKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
  static GEMMKernel kernel;

  bool transpose = true;
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  auto packw = kernel.getWeightPtr()->createStorage(n, k);
  auto pack_wei_tensor = torch::empty(packw.mSize, torch::kInt8);
  packw.assign(pack_wei_tensor.data_ptr<int8_t>());
  if (transpose) {
    kernel.getWeightPtr()->packWeightTranspose(n, k,
                                               {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), k, &packw});
  } else {
    kernel.getWeightPtr()->packWeight(n, k, {reinterpret_cast<jblas::utils::bf16*>(weight.data_ptr()), n, &packw});
  }

  TORCH_CHECK(activation.dim() == 2, "Qbits: only support 2-dim activation in mmbf16.");
  // auto deseries_wei = jblas::prologue::gemm::PackedWeightParser::deserialBuffer(weight.data_ptr());
  int m = output.sizes()[0];
  // int n = output.sizes()[1];
  // int k = activation.sizes()[1];
  GEMMKernel::Arguments args{m, n, k, reinterpret_cast<jblas::utils::bf16*>(activation.data_ptr()), k, NULL, 0,
                             //  reinterpret_cast<jblas::prologue::gemm::StoragePackedWeight*>(deseries_wei),
                             &packw, reinterpret_cast<jblas::utils::bf16*>(output.data_ptr()), n, NULL};
  kernel.compute(args);
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits mmbf16 verbose\nm:" << m << " n:" << n << " k:" << k << " execute time:" << cost_time << "ms";
  }
}

void jblas_mmfp32_avx2(torch::Tensor& activation, torch::Tensor& weight, torch::Tensor& output) {
  if (initer.verbose) timer.start();
  using GEMMKernel = jblas::wrapper::gemm_default::avx2::GemmKernel;
  static GEMMKernel kernel;

  bool transpose = true;
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  auto packw = kernel.getWeightPtr()->createStorage(n, k);
  auto pack_wei_tensor = torch::empty(packw.mSize, torch::kInt8);
  packw.assign(pack_wei_tensor.data_ptr<int8_t>());
  if (transpose) {
    kernel.getWeightPtr()->packWeightTranspose(n, k, {reinterpret_cast<float*>(weight.data_ptr()), k, &packw});
  } else {
    kernel.getWeightPtr()->packWeight(n, k, {reinterpret_cast<float*>(weight.data_ptr()), n, &packw});
  }

  TORCH_CHECK(activation.dim() == 2, "Qbits: only support 2-dim activation in mmbf16.");
  auto deseries_wei = jblas::prologue::gemm::PackedWeightParser::deserialBuffer(weight.data_ptr());
  int m = output.sizes()[0];
  // int n = output.sizes()[1];
  // int k = activation.sizes()[1];
  GEMMKernel::Arguments args{m, n, k, reinterpret_cast<float*>(activation.data_ptr()), k, NULL, 0,
                             //  reinterpret_cast<jblas::prologue::gemm::StoragePackedWeight*>(deseries_wei),
                             &packw, reinterpret_cast<float*>(output.data_ptr()), n, NULL};
  kernel.compute(args);
  if (initer.verbose) {
    timer.stop();
    auto cost_time = timer.get_elapsed_time();
    LOG(INFO) << "QBits mmfp32-avx2 verbose\nm:" << m << " n:" << n << " k:" << k << " execute time:" << cost_time
              << "ms";
  }
}