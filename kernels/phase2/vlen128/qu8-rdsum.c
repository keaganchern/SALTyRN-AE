#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(c) (c)
#endif

struct xnn_qs8_rsum_params;

/**
 * Optimized row sum for Saturn RISC-V Vector Microarchitecture.
 * Uses Decoupled Access-Execute (DAE) features and maximum LMUL for chime length.
 */
void test_rvv(
    size_t rows,
    size_t channels,
    const uint8_t* restrict input,
    size_t input_stride,
    const uint8_t* restrict zero,
    uint32_t* restrict output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  // Precompute scalar offsets to minimize address calculation overhead in the inner loop.
  const size_t s1 = input_stride;
  const size_t s2 = s1 + s1;
  const size_t s3 = s2 + s1;
  const size_t s4 = s3 + s1;
  const size_t s5 = s4 + s1;
  const size_t s6 = s5 + s1;
  const size_t row_7_increment = 7 * input_stride;

  while (channels > 0) {
    // Strip-mine channels using maximum LMUL (8) for 32-bit accumulators.
    // This maximizes chime length, helping to hide pipeline latencies and reduce scalar pressure.
    size_t vl = __riscv_vsetvl_e32m8(channels);
    
    // Load existing output values into the wide (e32m8) accumulator via the VLS sequencer.
    vuint32m8_t vacc = __riscv_vle32_v_u32m8(output, vl);

    const uint8_t* i = input;
    size_t r = rows;

    while (r > 0) {
      // Limit batch size to 254 rows to avoid 16-bit overflow (254 * 255 = 64770 < 65535).
      size_t current_batch = (r < 254) ? r : 254;
      r -= current_batch;

      // Begin the batch using a widening initialization.
      vuint8m2_t vin0 = __riscv_vle8_v_u8m2(i, vl);
      vuint16m4_t vacc16;

      if (current_batch >= 2) {
        // Optimization: Use Vector-Vector widening add to initialize the 16-bit intermediate accumulator.
        vuint8m2_t vin1 = __riscv_vle8_v_u8m2(i + s1, vl);
        vacc16 = __riscv_vwaddu_vv_u16m4(vin0, vin1, vl);
        i += s2;
        current_batch -= 2;
      } else {
        // Edge case: Handle a single row batch using zero-extension.
        vacc16 = __riscv_vzext_vf2_u16m4(vin0, vl);
        i += s1;
        current_batch -= 1;
      }

      // Fast path: Process 7 rows per iteration. 
      // We group Loads (VLS) followed by Adds (VXS). Saturn's decoupled sequencers allow 
      // the VLS to buffer multiple loads ahead while the VXS unit processes the adds.
      for (; current_batch >= 7; current_batch -= 7) {
        vuint8m2_t v0 = __riscv_vle8_v_u8m2(i, vl);
        vuint8m2_t v1 = __riscv_vle8_v_u8m2(i + s1, vl);
        vuint8m2_t v2 = __riscv_vle8_v_u8m2(i + s2, vl);
        vuint8m2_t v3 = __riscv_vle8_v_u8m2(i + s3, vl);
        vuint8m2_t v4 = __riscv_vle8_v_u8m2(i + s4, vl);
        vuint8m2_t v5 = __riscv_vle8_v_u8m2(i + s5, vl);
        vuint8m2_t v6 = __riscv_vle8_v_u8m2(i + s6, vl);

        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v0, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v1, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v2, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v3, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v4, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v5, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, v6, vl);

        i += row_7_increment;
      }

      // Handle the tail rows for the current batch strip.
      for (; current_batch > 0; current_batch--) {
        vuint8m2_t vin = __riscv_vle8_v_u8m2(i, vl);
        vacc16 = __riscv_vwaddu_wv_u16m4(vacc16, vin, vl);
        i += s1;
      }

      // Drain the 16-bit intermediate accumulator into the 32-bit destination accumulator.
      // Saturn's hardware decoupling overlaps this operation with the next batch's loads.
      vacc = __riscv_vwaddu_wv_u32m8(vacc, vacc16, vl);
    }

    // Store results back to memory using full memory bandwidth (unit-stride VSS).
    __riscv_vse32_v_u32m8(output, vacc, vl);

    // Advance channel pointers.
    input += vl;
    output += vl;
    channels -= vl;
  }
}
