#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

struct xnn_f32_default_params;

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

void test_rvv(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const size_t n = batch / sizeof(float);
  const float* restrict p_in_a = input_a;
  const float* restrict p_in_b = input_b;
  float* restrict p_out = output;

  size_t i = 0;
  const size_t vlmax = __riscv_vsetvlmax_e32m8();

  // Optimized Fast-Path: Process chunks using fixed vlmax to minimize scalar overhead (Strategy 6)
  if (__builtin_expect(n >= vlmax, 1)) {
    // Prologue: Load the first chunk
    vfloat32m8_t a0 = __riscv_vle32_v_f32m8(p_in_a, vlmax);
    vfloat32m8_t b0 = __riscv_vle32_v_f32m8(p_in_b, vlmax);
    p_in_a += vlmax;
    p_in_b += vlmax;
    i = vlmax;

    // Steady-state loop: No vsetvl inside
    while (i + vlmax <= n) {
      // Interleave memory and compute across VLS, VXS, and VSS sequencers
      vfloat32m8_t a1 = __riscv_vle32_v_f32m8(p_in_a, vlmax);    // VLS
      vfloat32m8_t vacc0 = __riscv_vfmul_vv_f32m8(a0, b0, vlmax); // VXS
      vfloat32m8_t b1 = __riscv_vle32_v_f32m8(p_in_b, vlmax);    // VLS
      __riscv_vse32_v_f32m8(p_out, vacc0, vlmax);               // VSS

      // Scalar bookkeeping (pointer and counter updates) overlaps with vector execution
      p_in_a += vlmax;
      p_in_b += vlmax;
      p_out += vlmax;
      i += vlmax;

      // Buffer rotation
      a0 = a1;
      b0 = b1;
    }

    // Epilogue for the fast-path: compute and store the final vlmax-chunk loaded
    vfloat32m8_t vacc_final = __riscv_vfmul_vv_f32m8(a0, b0, vlmax);
    __riscv_vse32_v_f32m8(p_out, vacc_final, vlmax);
    p_out += vlmax;
  }

  // Tail: Handle remaining elements using VLA logic
  if (i < n) {
    size_t vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t a_tail = __riscv_vle32_v_f32m8(p_in_a, vl);
    vfloat32m8_t b_tail = __riscv_vle32_v_f32m8(p_in_b, vl);
    vfloat32m8_t vacc_tail = __riscv_vfmul_vv_f32m8(a_tail, b_tail, vl);
    __riscv_vse32_v_f32m8(p_out, vacc_tail, vl);
  }
}
