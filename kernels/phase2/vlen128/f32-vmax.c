#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

struct xnn_f32_default_params;

/**
 * Saturn RVV Optimization: Strategy 13 (Memory Disambiguation via restrict).
 * 
 * Optimized to leverage Saturn's Decoupled Access-Execute (DAE) architecture.
 * By adding 'restrict', we allow the hardware's early-stage memory disambiguation
 * unit to verify independence between load/store streams (VLS/VSS) more efficiently.
 * This enables the sequencers to run ahead and maximizes the benefits of chaining.
 */
void test_rvv(
    size_t batch,
    const float* restrict input_a,
    const float* restrict input_b,
    float* restrict output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  
  // Strip-mining loop
  while (n > 0) {
    // Set VL for float32 elements with LMUL=8.
    // Saturn's short-vector design benefits from maximum LMUL to increase chime length.
    // A longer chime (LMUL * VLEN/DLEN) hides the 2-stage latency of the FPConvPipe
    // used by vfmax and reduces scalar instruction pressure.
    size_t vl = __riscv_vsetvl_e32m8(n);

    // Vector Unit-Stride Loads (processed by the VLS sequencer)
    // Unit-stride accesses operate at full memory bandwidth on Saturn.
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);

    // Compute element-wise maximum (processed by the VXS sequencer)
    // Chaining occurs between VLS and VXS as they occupy different sequencers.
    vfloat32m8_t vacc = __riscv_vfmax_vv_f32m8(va, vb, vl);

    // Vector Unit-Stride Store (processed by the VSS sequencer)
    // Chaining occurs between VXS and VSS.
    __riscv_vse32_v_f32m8(output, vacc, vl);

    // Scalar bookkeeping (pointer increments and loop counter)
    // Saturn's DAE allows the scalar core to execute these updates while the 
    // vector units (VU) are busy processing the current vector length (vl).
    // Note: No scalar stores are performed here, avoiding dispatch stalls.
    input_a += vl;
    input_b += vl;
    output += vl;
    n -= vl;
  }
}
