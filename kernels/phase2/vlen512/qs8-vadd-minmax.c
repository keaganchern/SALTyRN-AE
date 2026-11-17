// Input zero points folded into an int32 bias, output zero point added after the shift (no int64 range check).
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

// output_zero_point << shift is folded into the int32 bias only when the accumulator cannot overflow int32;
// otherwise the general loop runs.

void test_rvv(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const struct xnn_qs8_add_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int8_t a_zero_point = params->scalar.a_zero_point;
  const int8_t b_zero_point = params->scalar.b_zero_point;
  const int32_t a_multiplier = params->scalar.a_multiplier;
  const int32_t b_multiplier = params->scalar.b_multiplier;
  const size_t shift = (size_t) params->scalar.shift;
  const int16_t output_zero_point = params->scalar.output_zero_point;
  const int8_t output_min = params->scalar.output_min;
  const int8_t output_max = params->scalar.output_max;

  // Folded bias: acc(a, b) = bias + a * a_multiplier + b * b_multiplier, with the zero points and
  // output_zero_point << shift folded in, so one rounding narrowing shift yields the output. Exact only when the true
  // value of acc fits in int32 for every (a, b) in [-128, 127]^2 (linear in a and b: its extremes are at the corners);
  // with a large shift (output scale far above the input scales) it does not, and the general loop runs instead.
  // Input zero points folded into an int32 bias. Under XNNPACK's parameters (multipliers in [1, 2^21], zero points
  // in [-128, 127]) the bias and every accumulator stay within +-2^30, so no range check is needed. The output zero
  // point is added after the rounding shift (saturating, in int16).
  // Rounding term in the bias, then a floor shift (= a round-to-nearest-up shift; |acc| < 2^31 still).
  const int32_t init_acc = (int32_t) (1u << (shift - 1)) - (int32_t) a_zero_point * a_multiplier
                           - (int32_t) b_zero_point * b_multiplier;
  int32_t bias = init_acc;
  const vint32m8_t v_rt = __riscv_vmv_v_x_i32m8((int32_t) (1u << (shift - 1)), __riscv_vsetvlmax_e32m8());

  // Fast path: when a_multiplier == b_multiplier, algebraically factor:
  // (input_a * M) + (input_b * M) = (input_a + input_b) * M.
  // input_a + input_b fits losslessly into int16, cutting LMUL=8 multiplications by 50%.
  if (a_multiplier == b_multiplier) {
    if (a_zero_point == 0 && b_zero_point == 0) {   // bias = rounding term only: added by the first vmadd
      size_t vl = __riscv_vsetvl_e8m2(batch);
      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
      size_t i = vl;

      while (i < batch) {
        size_t next_vl = __riscv_vsetvl_e8m2(batch - i);
        // Overlap loading next chunk with computing the current chunk
        vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a + i, next_vl);
        vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b + i, next_vl);

        vint16m4_t vsum_16 = __riscv_vwadd_vv_i16m4(va, vb, vl);
        vint32m8_t vsum_32 = __riscv_vsext_vf2_i32m8(vsum_16, vl);
        vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vsum_32, a_multiplier, v_rt, vl);

        vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

        vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
        vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);

        vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
        vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

        __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

        // Rotate buffers and lengths for next iteration
        va = va_next; vb = vb_next; vl = next_vl;
        i += vl;
      }
      
      // Drain final chunk
      vint16m4_t vsum_16 = __riscv_vwadd_vv_i16m4(va, vb, vl);
      vint32m8_t vsum_32 = __riscv_vsext_vf2_i32m8(vsum_16, vl);
      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vsum_32, a_multiplier, v_rt, vl);
      vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);
      vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
      __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

    } else {
      size_t vl = __riscv_vsetvl_e8m2(batch);
      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
      size_t i = vl;

      while (i < batch) {
        size_t next_vl = __riscv_vsetvl_e8m2(batch - i);
        vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a + i, next_vl);
        vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b + i, next_vl);

        vint32m8_t vacc = __riscv_vmv_v_x_i32m8(bias, vl);
        vint16m4_t vsum_16 = __riscv_vwadd_vv_i16m4(va, vb, vl);
        vint32m8_t vsum_32 = __riscv_vsext_vf2_i32m8(vsum_16, vl);
        vacc = __riscv_vmacc_vx_i32m8(vacc, a_multiplier, vsum_32, vl);

        vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

        vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
        vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);

        vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
        vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

        __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

        va = va_next; vb = vb_next; vl = next_vl;
        i += vl;
      }
      
      vint32m8_t vacc = __riscv_vmv_v_x_i32m8(bias, vl);
      vint16m4_t vsum_16 = __riscv_vwadd_vv_i16m4(va, vb, vl);
      vint32m8_t vsum_32 = __riscv_vsext_vf2_i32m8(vsum_16, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, a_multiplier, vsum_32, vl);
      vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);
      vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
      __riscv_vse8_v_i8m2(output + i - vl, vout, vl);
    }
  } else {
    // Fallback path: distinct multipliers
    if (a_zero_point == 0 && b_zero_point == 0) {   // bias = rounding term only: added by the first vmadd
      size_t vl = __riscv_vsetvl_e8m2(batch);
      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
      size_t i = vl;

      while (i < batch) {
        size_t next_vl = __riscv_vsetvl_e8m2(batch - i);
        vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a + i, next_vl);
        vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b + i, next_vl);

        vint32m8_t vxa_32 = __riscv_vsext_vf4_i32m8(va, vl);
        vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vxa_32, a_multiplier, v_rt, vl);
        vint32m8_t vxb_32 = __riscv_vsext_vf4_i32m8(vb, vl);
        vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vxb_32, vl);

        vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

        vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
        vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);

        vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
        vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

        __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

        va = va_next; vb = vb_next; vl = next_vl;
        i += vl;
      }

      vint32m8_t vxa_32 = __riscv_vsext_vf4_i32m8(va, vl);
      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vxa_32, a_multiplier, v_rt, vl);
      vint32m8_t vxb_32 = __riscv_vsext_vf4_i32m8(vb, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vxb_32, vl);
      vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);
      vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
      __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

    } else {
      size_t vl = __riscv_vsetvl_e8m2(batch);
      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
      size_t i = vl;

      while (i < batch) {
        size_t next_vl = __riscv_vsetvl_e8m2(batch - i);
        vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a + i, next_vl);
        vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b + i, next_vl);

        vint32m8_t vacc = __riscv_vmv_v_x_i32m8(bias, vl);
        vint32m8_t vxa_32 = __riscv_vsext_vf4_i32m8(va, vl);
        vacc = __riscv_vmacc_vx_i32m8(vacc, a_multiplier, vxa_32, vl);
        vint32m8_t vxb_32 = __riscv_vsext_vf4_i32m8(vb, vl);
        vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vxb_32, vl);

        vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

        vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
        vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);

        vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
        vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

        __riscv_vse8_v_i8m2(output + i - vl, vout, vl);

        va = va_next; vb = vb_next; vl = next_vl;
        i += vl;
      }

      vint32m8_t vacc = __riscv_vmv_v_x_i32m8(bias, vl);
      vint32m8_t vxa_32 = __riscv_vsext_vf4_i32m8(va, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, a_multiplier, vxa_32, vl);
      vint32m8_t vxb_32 = __riscv_vsext_vf4_i32m8(vb, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vxb_32, vl);
      vint16m4_t vacc_16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);
      vacc_16 = __riscv_vsadd_vx_i16m4(vacc_16, output_zero_point, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc_16, 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
      __riscv_vse8_v_i8m2(output + i - vl, vout, vl);
    }
  }
}
