#include "../include/dropout.hpp"
#include <immintrin.h>

template <bool BF16>
static inline void write_rand(void* data, int thread_idx, int64_t elt_num, int dt_size, double p, void* mask_ptr) {
  int i = 0;
  auto zmm_scale = _mm512_set1_ps(1.f / (1.f - p));
  auto zmm_p = _mm512_set1_ps(float(p));
  int align_elt_num = elt_num / 16 * 16;
  for (; i < align_elt_num; i += 16) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(zmm_p, randv);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    if constexpr (!BF16) {
      auto ans = _mm512_loadu_ps(data + i * dt_size);
      ans = _mm512_mul_ps(ans, mul_scale);
      _mm512_storeu_ps(data + i * dt_size, ans);
      _mm512_storeu_ps(mask_ptr + i * dt_size, mul_scale);
    } else {
      auto ans = _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<float*>(data + i * dt_size)));
      ans = _mm512_mul_ps(ans, mul_scale);
      auto bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
      auto bf16_mul_scale = (__m256i)_mm512_cvtneps_pbh(mul_scale);
      _mm256_storeu_epi32(data + i * dt_size, bf16_ans);
      _mm256_storeu_epi32(mask_ptr + i * dt_size, bf16_mul_scale);
    }
    // _mm512_storeu_ps(mask_ptr + i * dt_size, mul_scale);
  }
  if (i < elt_num) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto ls_mask = _cvtu32_mask16(0xffff >> (16 - elt_num + i));
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(zmm_p, randv);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    if constexpr (!BF16) {
      __m512 ans;
      ans = _mm512_mask_loadu_ps(ans, ls_mask, data + i * dt_size);
      ans = _mm512_mul_ps(ans, mul_scale);
      _mm512_mask_storeu_ps(data + i * dt_size, ls_mask, ans);
      _mm512_mask_storeu_ps(mask_ptr + i * dt_size, ls_mask, mul_scale);
    } else {
      auto ans = _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(reinterpret_cast<float*>(data + i * dt_size)));
      ans = _mm512_mul_ps(ans, mul_scale);
      auto bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
      auto bf16_mul_scale = (__m256i)_mm512_cvtneps_pbh(mul_scale);
      _mm256_mask_storeu_epi32(data + i * dt_size, ls_mask, bf16_ans);
      _mm256_mask_storeu_epi32(mask_ptr + i * dt_size, ls_mask, bf16_mul_scale);
    }
    // _mm512_mask_storeu_ps(mask_ptr + i * dt_size, ls_mask, mul_scale);
  }
}

torch::Tensor dropout(torch::Tensor& output, double p) {
  auto elt_num = output.numel();
  auto core_num = omp_get_max_threads();
  auto task_each_core = elt_num / core_num;
  torch::Tensor mask = torch::empty_like(output);
#pragma omp parallel
  {
    auto ker_idx = omp_get_thread_num();
    auto tasks = ker_idx == (core_num - 1) ? elt_num - (core_num - 1) * task_each_core : task_each_core;
    if (output.scalar_type() == torch::kFloat32) {
      write_rand<false>(output.data_ptr() + ker_idx * task_each_core * output.element_size(), ker_idx, tasks,
                        output.element_size(), p, mask.data_ptr() + ker_idx * task_each_core * output.element_size());
    } else if (output.scalar_type() == torch::kBFloat16) {
      write_rand<true>(output.data_ptr() + ker_idx * task_each_core * output.element_size(), ker_idx, tasks,
                       output.element_size(), p, mask.data_ptr() + ker_idx * task_each_core * output.element_size());
    } else {
      TORCH_CHECK(false, "Qbits: unsupported input data type in dropout operator.");
    }
  }
  return mask;
}