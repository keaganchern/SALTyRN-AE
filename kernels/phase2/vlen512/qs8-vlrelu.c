#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int8_t input_zero_point = (int8_t) params->scalar.input_zero_point;
  // Pre-shift the input zero point by 7 (equivalent to multiplying by 128)
  const int16_t shifted_izp = (int16_t)input_zero_point << 7; 
  // negated: 32768 does not fit int16
  const int16_t positive_multiplier = (int16_t) (-params->scalar.positive_multiplier);
  // negated: 32768 does not fit int16
  const int16_t negative_multiplier = (int16_t) (-params->scalar.negative_multiplier);
  const int16_t output_zero_point = (int16_t) params->scalar.output_zero_point;

  // Prime the pipeline: load the first chunk of data
  size_t vl = __riscv_vsetvl_e8m4(batch);

  // Explicit software prefetching for the first few chunks
  // Prefetch current chunk and the next chunk
  __builtin_prefetch(input, 0, 3);
  __builtin_prefetch(output, 1, 3);
  __builtin_prefetch(input + vl, 0, 3);
  __builtin_prefetch(output + vl, 1, 3);

  vint8m4_t vx = __riscv_vle8_v_i8m4(input, vl);
  input += vl;
  batch -= vl;

  while (batch > 0) {
    // Consume current vx for mask
    const vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx, input_zero_point, vl);
    // Multiply by 128 directly using MulPipe; issues in parallel with vmslt (IntPipe)
    vint16m8_t vx_128 = __riscv_vwmulsu_vx_i16m8(vx, 128, vl);

    // Overlap data movement and compute:
    // With vx consumed, issue the load for the next iteration to VLS immediately.
    // Saturn's VLS fetches next_vx in parallel while VXS computes the rest of this iteration.
    const size_t next_vl = __riscv_vsetvl_e8m4(batch);

    // Explicit software prefetching for future iterations to warm up backing memory
    // 'input' has already advanced, so 'input + next_vl' points to the iteration after next.
    // We do the equivalent for output: 'output + vl + next_vl'.
    __builtin_prefetch(input + next_vl, 0, 3);
    __builtin_prefetch(output + vl + next_vl, 1, 3);

    const vint8m4_t next_vx = __riscv_vle8_v_i8m4(input, next_vl);
    input += next_vl;
    batch -= next_vl;

    // (izp - x) * 128: with the negated multipliers the product is (x - izp) * 128 * multiplier
    vint16m8_t vacc = __riscv_vrsub_vx_i16m8(vx_128, shifted_izp, vl);
    
    // Replace vmerge + vsmul.vv with an unmasked vsmul.vx + masked vsmul.vx_mu
    vint16m8_t vacc_mult = __riscv_vsmul_vx_i16m8(vacc, positive_multiplier, __RISCV_VXRM_RNU, vl);
    vacc = __riscv_vsmul_vx_i16m8_mu(vmask, vacc_mult, vacc, negative_multiplier, __RISCV_VXRM_RNU, vl);
    
    vacc = __riscv_vsadd_vx_i16m8(vacc, output_zero_point, vl);

    const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m4(output, vy, vl);
    output += vl;

    vx = next_vx;
    vl = next_vl;
  }

  // Tail iteration: process the final chunk
  const vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx, input_zero_point, vl);
  vint16m8_t vx_128 = __riscv_vwmulsu_vx_i16m8(vx, 128, vl);
  vint16m8_t vacc = __riscv_vrsub_vx_i16m8(vx_128, shifted_izp, vl);
  
  vint16m8_t vacc_mult = __riscv_vsmul_vx_i16m8(vacc, positive_multiplier, __RISCV_VXRM_RNU, vl);
  vacc = __riscv_vsmul_vx_i16m8_mu(vmask, vacc_mult, vacc, negative_multiplier, __RISCV_VXRM_RNU, vl);
  
  vacc = __riscv_vsadd_vx_i16m8(vacc, output_zero_point, vl);
  const vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
  __riscv_vse8_v_i8m4(output, vy, vl);
}
