#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

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

  size_t n = batch / sizeof(float);
  
  // Obtain the hardware's maximum vector length for e32m8 once.
  // Using LMUL=8 maximizes the chime length on Saturn's short-vector architecture,
  // which helps hide pipeline latencies (like the 4-stage FMA) and reduces 
  // scalar instruction fetch pressure.
  const size_t vlmax = __riscv_vsetvlmax_e32m8();

  // Unified loop: process the full data array and merge the speculative tail
  // handling into the final iteration to eliminate code duplication entirely.
  // The loop condition and increment use vlmax to maintain alignment with the 
  // hardware's element-group processing (DLEN).
  // Full vectors while at least vlmax elements remain, then the tail with vl = n, so no
  // load reads past the end of the inputs.
  for (; n >= vlmax; n -= vlmax) {
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vlmax);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vlmax);
    vfloat32m8_t vacc = __riscv_vfsub_vv_f32m8(va, vb, vlmax);
    __riscv_vse32_v_f32m8(output, vacc, vlmax);
    input_a += vlmax;
    input_b += vlmax;
    output += vlmax;
  }
  if (n > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    vfloat32m8_t vacc = __riscv_vfsub_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
  }
}
