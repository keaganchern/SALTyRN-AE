#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

// Define the struct locally for completeness, assuming standard definition.
struct xnn_qs8_rsum_params;

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

void test_rvv(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  // Pre-calculate stride multiples to reduce scalar ALU operations within the unrolled loop.
  const size_t s1 = input_stride;
  const size_t s2 = s1 * 2;
  const size_t s3 = s1 * 3;
  const size_t s4 = s1 * 4;

  while (channels > 0) {
    // Strategy: Maximize LMUL to 4 for the 8-bit inputs.
    // Saturn benefits from LMUL=4 (chime length ~4) to hide pipeline latencies.
    size_t vl = __riscv_vsetvl_e8m4(channels);

    const int8_t* i = input;
    size_t r = rows;

    while (r > 0) {
      // Accumulate up to 252 rows into a 16-bit vector (LMUL=8) to prevent signed overflow.
      // (252 * 127 = 32004 < 32768).
      size_t batch = (r < 252) ? r : 252;
      r -= batch;

      // Strategy 11: Zero-Initialization Splats.
      // Initializing the 16-bit accumulator to zero once per batch avoids the 
      // RAW dependency and SEW-switch overhead of sign-extending the first row.
      // Note: We use __riscv_vmv_v_x_i16m8 instead of __riscv_vmv_v_i_i16m8 for standard compliance.
      vint16m8_t vacc16 = __riscv_vmv_v_x_i16m8(0, vl);

      // Process rows in the current batch.
      // Saturn's DAE hardware chaining overlaps unit-stride loads (VLS) and widening adds (VXS).
      while (batch >= 4) {
        vint8m4_t vin0 = __riscv_vle8_v_i8m4(i, vl);
        vacc16 = __riscv_vwadd_wv_i16m8(vacc16, vin0, vl);
        
        vint8m4_t vin1 = __riscv_vle8_v_i8m4(i + s1, vl);
        vacc16 = __riscv_vwadd_wv_i16m8(vacc16, vin1, vl);
        
        vint8m4_t vin2 = __riscv_vle8_v_i8m4(i + s2, vl);
        vacc16 = __riscv_vwadd_wv_i16m8(vacc16, vin2, vl);
        
        vint8m4_t vin3 = __riscv_vle8_v_i8m4(i + s3, vl);
        vacc16 = __riscv_vwadd_wv_i16m8(vacc16, vin3, vl);
        
        i = (const int8_t*) ((uintptr_t) i + s4);
        batch -= 4;
      }

      while (batch > 0) {
        vint8m4_t vin = __riscv_vle8_v_i8m4(i, vl);
        vacc16 = __riscv_vwadd_wv_i16m8(vacc16, vin, vl);
        i = (const int8_t*) ((uintptr_t) i + s1);
        batch -= 1;
      }

      // Update the 32-bit output. Since e32m16 is illegal, we split the e16m8 
      // accumulator into two e16m4 halves to maintain alignment with the e32m8 output.
      size_t vl_lo = __riscv_vsetvl_e32m8(vl);
      vint16m4_t vacc16_lo = __riscv_vget_v_i16m8_i16m4(vacc16, 0);
      
      vint32m8_t out_lo = __riscv_vle32_v_i32m8(output, vl_lo);
      out_lo = __riscv_vwadd_wv_i32m8(out_lo, vacc16_lo, vl_lo);
      __riscv_vse32_v_i32m8(output, out_lo, vl_lo);

      size_t vl_hi = vl - vl_lo;
      if (vl_hi > 0) {
        vint16m4_t vacc16_hi = __riscv_vget_v_i16m8_i16m4(vacc16, 1);
        
        vint32m8_t out_hi = __riscv_vle32_v_i32m8(output + vl_lo, vl_hi);
        out_hi = __riscv_vwadd_wv_i32m8(out_hi, vacc16_hi, vl_hi);
        __riscv_vse32_v_i32m8(output + vl_lo, out_hi, vl_hi);
      }
    }

    // Advance to the next strip of channels.
    input += vl;
    output += vl;
    channels -= vl;
  }
}
