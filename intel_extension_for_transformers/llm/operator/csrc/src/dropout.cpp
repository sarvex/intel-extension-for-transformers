#include "../include/dropout.hpp"
#include <immintrin.h>

static inline void write_rand(void* data, int thread_idx, int64_t elt_num, int dt_size, double p, torch::Tensor& mask) {
  int i = 0;
  assert(dt_size == 4);
  auto zmm_scale = _mm512_set1_ps(1.f / (1.f - p));
  auto zmm_p = _mm512_set1_ps(p);
  int align_elt_num = elt_num / 16 * 16;
  auto mask_ptr = mask.data_ptr();
  for (; i < align_elt_num; i += 16) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(randv, zmm_p);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    auto ans = _mm512_loadu_ps(data + i * dt_size);
    ans = _mm512_mul_ps(ans, mul_scale);
    _mm512_storeu_ps(data + i * dt_size, ans);
    _mm512_storeu_ps(mask_ptr + i * dt_size, mul_scale);
  }
  if (i < elt_num) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto ls_mask = _cvtu32_mask16(0xffff >> (16 - elt_num + i));
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(randv, zmm_p);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    __m512 ans;
    ans = _mm512_mask_loadu_ps(ans, ls_mask, data + i * dt_size);
    ans = _mm512_mul_ps(ans, mul_scale);
    _mm512_mask_storeu_ps(data + i * dt_size, ls_mask, ans);
    _mm512_mask_storeu_ps(mask_ptr + i * dt_size, ls_mask, mul_scale);
  }
}

torch::Tensor dropout(torch::Tensor& output, double p) {
  auto elt_num = output.numel();
  auto core_num = omp_get_max_threads();
  auto task_each_core = elt_num / core_num;
  torch::Tensor mask = torch::empty_like(output);
#pragma omp parallel
  // for (int i = 0; i < core_num; i++)
  {
    auto ker_idx = omp_get_thread_num();
    // auto ker_idx = i;
    auto tasks = ker_idx == (core_num - 1) ? elt_num - (core_num - 1) * task_each_core : task_each_core;
    // std::cout << "core:" << ker_idx << " tasks:" << tasks << std::endl;
    write_rand(output.data_ptr() + ker_idx * task_each_core * output.element_size(), ker_idx, tasks,
               output.element_size(), p, mask);
  }
  return mask;
}