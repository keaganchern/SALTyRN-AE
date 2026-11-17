#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  // Saturn SEW/LMUL configuration
  // e8m4: 4 registers for input (allows a long chime length to hide pipeline latencies)
  // e16m8: 8 registers for intermediate sum (preserves exact 1:1 element mapping with e8m4)
  // e32m8: 8 registers for global sum
  const size_t vlmax8 = __riscv_vsetvlmax_e8m4();
  const size_t vlmax32 = __riscv_vsetvlmax_e32m8();
  const size_t stride4 = 4 * vlmax8;
  
  // A 16-bit lane can safely accumulate up to 256 8-bit values (256 * 255 = 65280 < 65535), but the two halves of
  // the 16-bit accumulator are then added in 16 bits before widening, so a lane takes at most 128 values per
  // segment (2 * 128 * 255 = 65280 < 65535).
  const size_t max_segment_len = 128 * vlmax8;
  const uint8_t* end = input + batch;

  // Global 32-bit accumulator.
  vuint32m8_t vacc32 = __riscv_vmv_v_x_u32m8(0, vlmax32);

  while (input < end) {
    // 16-bit intermediate accumulator (LMUL=8) to maximize chime length.
    vuint16m8_t vacc16 = __riscv_vmv_v_x_u16m8(0, vlmax8);
    
    // Define boundaries for the 16-bit accumulation segment to prevent overflow.
    size_t remaining = (size_t)(end - input);
    const uint8_t* inner_end = (remaining > max_segment_len) ? (input + max_segment_len) : end;
    
    // Hot loop: Interleave VLS (Loads) and VXS (Arithmetic) to saturate sequencers.
    while (input + stride4 <= inner_end) {
      vuint8m4_t v0 = __riscv_vle8_v_u8m4(input, vlmax8);
      vuint8m4_t v1 = __riscv_vle8_v_u8m4(input + vlmax8, vlmax8);
      vacc16 = __riscv_vwaddu_wv_u16m8(vacc16, v0, vlmax8);
      
      vuint8m4_t v2 = __riscv_vle8_v_u8m4(input + 2 * vlmax8, vlmax8);
      vacc16 = __riscv_vwaddu_wv_u16m8(vacc16, v1, vlmax8);
      
      vuint8m4_t v3 = __riscv_vle8_v_u8m4(input + 3 * vlmax8, vlmax8);
      vacc16 = __riscv_vwaddu_wv_u16m8(vacc16, v2, vlmax8);
      
      vacc16 = __riscv_vwaddu_wv_u16m8(vacc16, v3, vlmax8);
      input += stride4;
    }
    
    // Handle remaining full blocks within the segment.
    while (input + vlmax8 <= inner_end) {
      vuint8m4_t v0 = __riscv_vle8_v_u8m4(input, vlmax8);
      vacc16 = __riscv_vwaddu_wv_u16m8(vacc16, v0, vlmax8);
      input += vlmax8;
    }

    // VLA tail for the segment: Use tail-undisturbed policy to keep existing accumulated sums 
    // in elements past 'vl' so they can still be safely horizontally summed later.
    if (input < inner_end) {
      size_t vl = __riscv_vsetvl_e8m4((size_t)(inner_end - input));
      vuint8m4_t vt = __riscv_vle8_v_u8m4(input, vl);
      vacc16 = __riscv_vwaddu_wv_u16m8_tu(vacc16, vacc16, vt, vl);
      input += vl;
    }
    
    // Reduce e16m8 intermediate (N elements) to e32m8 global (N/2 elements):
    // 1. Split m8 into two m4 halves and sum them element-wise to match the capacity of e32m8.
    vuint16m4_t part0 = __riscv_vget_v_u16m8_u16m4(vacc16, 0);
    vuint16m4_t part1 = __riscv_vget_v_u16m8_u16m4(vacc16, 1);
    vuint16m4_t sum16 = __riscv_vadd_vv_u16m4(part0, part1, vlmax32);
    
    // 2. Widen the summed 16-bit values into the 32-bit global accumulator.
    vacc32 = __riscv_vwaddu_wv_u32m8(vacc32, sum16, vlmax32);
  }

  // Final horizontal reduction.
  // Initialize the horizontal sum with the existing value in *output to avoid a separate 
  // scalar addition which causes a RAW hazard stall after the vector-to-scalar writeback.
  vuint32m1_t v_init = __riscv_vmv_s_x_u32m1(*output, __riscv_vsetvlmax_e32m1());
  vuint32m1_t red_vec = __riscv_vredsum_vs_u32m8_u32m1(vacc32, v_init, vlmax32);
  
  // Single scalar writeback strictly outside the loop.
  *output = __riscv_vmv_x_s_u32m1_u32(red_vec);
}
