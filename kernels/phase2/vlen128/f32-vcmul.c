#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

struct xnn_f32_default_params;

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

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);

  size_t n = batch / sizeof(float);
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m4(n);

    // Strategy 20: Software Prefetching
    // Prefetch input data for future iterations to hide memory latency.
    // Saturn's decoupled scalar core issues these ahead of vector execution.
    // A distance of 64 elements (256 bytes) provides a robust lookahead.
    __builtin_prefetch(ar + 64, 0, 3);
    __builtin_prefetch(ai + 64, 0, 3);
    __builtin_prefetch(br + 64, 0, 3);
    __builtin_prefetch(bi + 64, 0, 3);

    // Interleave loads, arithmetic, and scalar bookkeeping to maximize chaining 
    // across the VLS (Load), VXS (Execute), and VSS (Store) sequencers.
    
    vfloat32m4_t var = __riscv_vle32_v_f32m4(ar, vl);
    ar += vl;
    
    vfloat32m4_t vbr = __riscv_vle32_v_f32m4(br, vl);
    br += vl;
    
    // VXS arithmetic chains with the VLS loads of var and vbr
    vfloat32m4_t vaccr = __riscv_vfmul_vv_f32m4(var, vbr, vl);

    vfloat32m4_t vbi = __riscv_vle32_v_f32m4(bi, vl);
    bi += vl;
    
    // VXS arithmetic chains with the VLS load of vbi
    vfloat32m4_t vacci = __riscv_vfmul_vv_f32m4(var, vbi, vl);

    vfloat32m4_t vai = __riscv_vle32_v_f32m4(ai, vl);
    ai += vl;
    
    // VXS arithmetic chains with the VLS load of vai
    vfloat32m4_t temp_r = __riscv_vfmul_vv_f32m4(vai, vbi, vl);
    
    // Use the maximum LMUL=4 to hide the 4-stage FMA/arithmetic latency
    vaccr = __riscv_vfsub_vv_f32m4(vaccr, temp_r, vl);
    
    // VSS store chains with the VXS subtraction
    __riscv_vse32_v_f32m4(or, vaccr, vl);
    or += vl;

    vfloat32m4_t temp_i = __riscv_vfmul_vv_f32m4(vai, vbr, vl);
    vacci = __riscv_vfadd_vv_f32m4(vacci, temp_i, vl);
    
    // VSS store chains with the VXS addition
    __riscv_vse32_v_f32m4(oi, vacci, vl);
    oi += vl;

    n -= vl;
  }
}
