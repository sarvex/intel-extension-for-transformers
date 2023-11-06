#include "../include/dropout.hpp"

static inline void write_rand(void* data, int thread_idx, int64_t elt_num, int dt_size) {
  int i = 0;
  assert(dt_size == 4);
  int align_elt_num = elt_num / 16 * 16;
  for (; i < align_elt_num; i += 16) {
    auto ans = rand_generator.gen_randfp(thread_idx);
    _mm512_storeu_ps(data + i * dt_size, ans);
  }
  if (i < elt_num) {
    auto ans = rand_generator.gen_randfp(thread_idx);
    auto mask = _cvtu32_mask16(0xffff >> (16 - elt_num + i));
    _mm512_mask_storeu_ps(data + i * dt_size, mask, ans);
  }
}

torch::Tensor dropout(torch::Tensor& output) {
  auto elt_num = output.numel();
  auto core_num = omp_get_max_threads();
  auto task_each_core = elt_num / core_num;
  torch::Tensor mask = torch::empty_like(output, torch::kInt8);
#pragma omp parallel
  // for (int i = 0; i < core_num; i++)
  {
    auto ker_idx = omp_get_thread_num();
    // auto ker_idx = i;
    auto tasks = ker_idx == (core_num - 1) ? elt_num - (core_num - 1) * task_each_core : task_each_core;
    // std::cout << "core:" << ker_idx << " tasks:" << tasks << std::endl;
    write_rand(output.data_ptr() + ker_idx * task_each_core * output.element_size(), ker_idx, tasks,
               output.element_size());
  }
  return mask;
}