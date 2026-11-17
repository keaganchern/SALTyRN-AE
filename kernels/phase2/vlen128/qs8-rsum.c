#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

void test_rvv(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  // Pre-calculate fixed VLMAX values for different groupings.
  // In the Saturn microarchitecture, using the largest LMUL (m8) for the main accumulator
  // reduces scalar overhead and maximizes chime length, helping hide pipeline latencies.
  const size_t vlmax32 = __riscv_vsetvlmax_e32m8();
  const size_t vlmax8 = __riscv_vsetvlmax_e8m8();
  
  // Initialize the 32-bit main accumulator with zeros.
  vint32m8_t vacc32 = __riscv_vmv_v_x_i32m8(0, vlmax32);

  while (batch > 0) {
    size_t inner_batch = batch;
    // Accumulate up to 256 int8 values into a 16-bit chunk to avoid overflow.
    // (127 * 256 = 32512, which fits in signed int16).
    const size_t max_inner = 256 * vlmax32;
    if (inner_batch > max_inner) {
      inner_batch = max_inner;
    }
    batch -= inner_batch;
    
    // Initialize the 16-bit partial accumulator with zeros.
    // LMUL=4 is used here because it will be widened to LMUL=8 in the final step of the chunk.
    vint16m4_t vacc16 = __riscv_vmv_v_x_i16m4(0, vlmax32);
    
    // Maximized LMUL Hot Loop: Process one m8 register group (vlmax8 elements) per load.
    // Saturn's VLS sequencer handles unit-stride loads at full bandwidth.
    while (inner_batch >= vlmax8) {
      vint8m8_t vt_all = __riscv_vle8_v_i8m8(input, vlmax8);
      
      // Perform widening additions using m2 chunks of the m8 load into the m4 partial accumulator.
      // Chaining occurs between the VLS (load) and VXS (execute) sequencers.
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, __riscv_vget_v_i8m8_i8m2(vt_all, 0), vlmax32);
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, __riscv_vget_v_i8m8_i8m2(vt_all, 1), vlmax32);
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, __riscv_vget_v_i8m8_i8m2(vt_all, 2), vlmax32);
      vacc16 = __riscv_vwadd_wv_i16m4(vacc16, __riscv_vget_v_i8m8_i8m2(vt_all, 3), vlmax32);

      input += vlmax8;
      inner_batch -= vlmax8;
    }

    // Handle remaining elements within the current 16-bit chunk.
    while (inner_batch > 0) {
      size_t vl = __riscv_vsetvl_e8m2(inner_batch);
      vint8m2_t vt = __riscv_vle8_v_i8m2(input, vl);
      // Policy _tu (tail-undisturbed) is used to preserve partial sums in inactive elements
      // when the vector length is shorter than vlmax32.
      vacc16 = __riscv_vwadd_wv_i16m4_tu(vacc16, vacc16, vt, vl);
      
      input += vl;
      inner_batch -= vl;
    }
    
    // Accumulate the 16-bit wide-element vector into the 32-bit main accumulator.
    vacc32 = __riscv_vwadd_wv_i32m8(vacc32, vacc16, vlmax32);
  }

  // Strategy 20: Hierarchical Reduction to reduce horizontal reduction chime-length.
  // Replacing a single LMUL=8 vredsum (on the VPS sequencer) with a tree of vadd.vv 
  // instructions (on the VXS sequencer) is more efficient. VXS is pipelined, and 
  // reducing the register grouping before the final horizontal reduction minimizes 
  // the time the VPS unit is occupied.

  // Fold m8 -> m4
  const size_t vl4 = __riscv_vsetvlmax_e32m4();
  vint32m4_t vacc32_m4 = __riscv_vadd_vv_i32m4(__riscv_vget_v_i32m8_i32m4(vacc32, 0), 
                                              __riscv_vget_v_i32m8_i32m4(vacc32, 1), vl4);
  // Fold m4 -> m2
  const size_t vl2 = __riscv_vsetvlmax_e32m2();
  vint32m2_t vacc32_m2 = __riscv_vadd_vv_i32m2(__riscv_vget_v_i32m4_i32m2(vacc32_m4, 0), 
                                              __riscv_vget_v_i32m4_i32m2(vacc32_m4, 1), vl2);
  // Fold m2 -> m1
  const size_t vl1 = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc32_m1 = __riscv_vadd_vv_i32m1(__riscv_vget_v_i32m2_i32m1(vacc32_m2, 0), 
                                              __riscv_vget_v_i32m2_i32m1(vacc32_m2, 1), vl1);

  // Perform the final horizontal reduction on the resulting m1 register group.
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vl1);
  vint32m1_t vres = __riscv_vredsum_vs_i32m1_i32m1(vacc32_m1, vzero, vl1);
  
  // Extract final scalar sum via vector-to-scalar writeback.
  // This causes a scalar pipeline stall on Saturn and is kept outside the loops.
  int32_t sum = __riscv_vmv_x_s_i32m1_i32(vres);

  *output += sum;
}
