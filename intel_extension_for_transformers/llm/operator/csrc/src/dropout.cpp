#include "../include/dropout.hpp"

static inline void write_rand(void* data, int thread_idx, int64_t elt_num, int dt_size) {
  assert(elt_num % 16 == 0);
  //   auto zmm_one = _mm512_set1_epi32(0x3F800000);
  for (int i = 0; i < elt_num; i += 16) {
    auto ans = rand_generator.gen_randfp(thread_idx);
    _mm512_storeu_ps(data + i * dt_size, ans);
  }
}

void dropout(torch::Tensor& output) {
  auto elt_num = output.numel();
  auto core_num = omp_get_max_threads();
  // int core_num = 10;
  auto task_each_core = elt_num / core_num;
// std::cout << "==" << task_each_core << "==" << std::endl;
#pragma omp parallel
  {
    // for (int i = 0; i < 10; i++) {
    auto ker_idx = omp_get_thread_num();
    // auto ker_idx = i;
    auto tasks = ker_idx == (core_num - 1) ? elt_num - (core_num - 1) * task_each_core : task_each_core;
    std::cout << "core" << ker_idx << " tasks:" << tasks << " eltsize" << output.element_size() << std::endl;
    write_rand(output.data_ptr() + ker_idx * task_each_core * output.element_size(), ker_idx, tasks,
               output.element_size());
  }
}