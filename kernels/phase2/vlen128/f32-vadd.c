#include <riscv_vector.h>
#include <stddef.h>
#include <assert.h>

// Structure and macro definitions as per the original environment
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

  size_t n = batch / sizeof(float);
  if (n == 0) return;

  // Strategy 15: Static-VL Main Loop Partitioning
  // Obtain the maximum vector length for the given SEW/LMUL outside the loop.
  // This constant value allows the compiler and hardware to avoid vsetvl bubbles.
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  size_t i = 0;

  // --- Fast Path: Process chunks of VLMAX elements ---
  // We use a double-buffered approach to maximize concurrent sequencer utilization.
  if (n >= vlmax) {
    // Prologue: Load the first buffer (VLS Sequencer)
    vfloat32m8_t va0 = __riscv_vle32_v_f32m8(input_a, vlmax);
    vfloat32m8_t vb0 = __riscv_vle32_v_f32m8(input_b, vlmax);
    i = vlmax;

    // Main Loop: Operations are interleaved to keep VLS, VXS, and VSS sequencers busy.
    while (n - i >= vlmax) {
      // 1. Compute current chunk (VXS Sequencer)
      // LMUL=8 ensures a long chime (VLEN*8/DLEN), hiding the 4-stage FMA latency.
      vfloat32m8_t vacc0 = __riscv_vfadd_vv_f32m8(va0, vb0, vlmax);

      // Strategy 21: VLS/VSS Sequencer Interleaving
      // We interleave loads and stores to ensure that a store is never immediately
      // followed by multiple loads. This breaks up VLS issue queue pressure and 
      // balances traffic across the Load (VLS) and Store (VSS) sequencers.
      
      // 2. Load first half of next chunk (VLS Sequencer)
      vfloat32m8_t va1 = __riscv_vle32_v_f32m8(input_a + i, vlmax);

      // 3. Store current chunk (VSS Sequencer)
      // This chains directly from the VXS sequencer output. 
      // By placing it between the two loads, we allow the scalar core to 
      // alternate between VLS and VSS issue queues.
      __riscv_vse32_v_f32m8(output + i - vlmax, vacc0, vlmax);

      // 4. Load second half of next chunk (VLS Sequencer)
      vfloat32m8_t vb1 = __riscv_vle32_v_f32m8(input_b + i, vlmax);

      // 5. Update state
      // Use standard assignment for whole-vector register contents (Rule 9).
      va0 = va1;
      vb0 = vb1;
      i += vlmax;
    }

    // Epilogue: Process the final buffer loaded during the last iteration of the fast path
    vfloat32m8_t vacc_final = __riscv_vfadd_vv_f32m8(va0, vb0, vlmax);
    __riscv_vse32_v_f32m8(output + i - vlmax, vacc_final, vlmax);
  }

  // --- Slow Path: Tail Block ---
  // Processes any remaining elements (n < vlmax or the remainder of n/vlmax).
  if (i < n) {
    size_t vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a + i, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b + i, vl);
    vfloat32m8_t vacc = __riscv_vfadd_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output + i, vacc, vl);
  }
}
