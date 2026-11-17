#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

void test_rvv(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  // Saturn is a short-vector microarchitecture. We establish vlmax for the
  // SEW/LMUL ratio of 4 (e8m2, e16m4, e32m8).
  const size_t vlmax = __riscv_vsetvlmax_e8m2();
  
  // Initialize the final 32-bit parallel accumulators. LMUL=8 is used to
  // maximize occupancy and hide pipeline latencies.
  vint32m8_t vacc32 = __riscv_vmv_v_x_i32m8(0, vlmax);

  // Strategy 7: Implement a dual-loop structure.
  // Precompute the block size for the fast-path loop to prevent redundant scalar math.
  const size_t batch_block = 128 * vlmax;

  // 1. Fast-path loop: Processes full 128-iteration blocks of size `vlmax`.
  // Eliminates `vsetvl` in the inner loop to avoid scalar frontend stalls/bubbles.
  while (batch >= batch_block) {
    // Tier 1 Accumulator (LMUL=4): Zeroed at the start of every block.
    vint16m4_t vacc16 = __riscv_vmv_v_x_i16m4(0, vlmax);
    
    // Fixed iteration count; no dynamic vl calculation required.
    for (size_t i = 0; i < 128; ++i) {
      // Load vector with constant vlmax: VLS sequencer.
      vint8m2_t vt8 = __riscv_vle8_v_i8m2(input, vlmax);
      
      input += vlmax;
      
      // Widening addition (Tier 1): int8 -> int16. VXS sequencer chains from VLS.
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, vt8, vlmax);
    }
    
    batch -= batch_block;

    // Tier 2: Promote Tier 1 sums (int16) to the final 32-bit parallel accumulator.
    vacc32 = __riscv_vwadd_wv_i32m8(vacc32, vacc16, vlmax);
  }

  // 2. Separate tail-handling block: Handles any remainder where batch < 128 * vlmax.
  // Because batch is already bounded, this loop will naturally execute at most 128 times, 
  // safely accumulating into int16_t without the risk of overflowing.
  if (batch > 0) {
    vint16m4_t vacc16 = __riscv_vmv_v_x_i16m4(0, vlmax);
    
    while (batch > 0) {
      // Dynamically calculate vl for the tail.
      size_t vl = __riscv_vsetvl_e8m2(batch);
      
      vint8m2_t vt8 = __riscv_vle8_v_i8m2(input, vl);
      
      input += vl;
      batch -= vl;
      
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, vt8, vl);
    }

    // Accumulate the tail using vlmax to prevent uninitialized element contamination.
    vacc32 = __riscv_vwadd_wv_i32m8(vacc32, vacc16, vlmax);
  }

  // Final horizontal reduction: Reduce the parallel 32-bit accumulators to a single scalar.
  // We use an LMUL=1 temporary for the reduction destination to minimize register usage.
  vint32m1_t vres = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vres = __riscv_vredsum_vs_i32m8_i32m1(vacc32, vres, vlmax);
  
  // Vector-to-scalar writeback (vmv.x.s) is performed outside the loop to avoid
  // stalling the scalar pipeline due to the post-commit execution nature of Saturn.
  *output += __riscv_vmv_x_s_i32m1_i32(vres);
}
