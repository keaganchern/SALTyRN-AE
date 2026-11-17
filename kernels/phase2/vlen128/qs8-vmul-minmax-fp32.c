#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

// Forward declaration of XNNPACK params struct
union xnn_qs8_mul_minmax_params {
  struct {
    int8_t a_zero_point;
    int8_t b_zero_point;
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
};

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

/**
 * Saturn RVV Optimization: xnn_qs8_mul_minmax
 * 
 * Strategy 13: Zero-Point Specialization.
 * In quantized multiplication: out = ((in_a - a_zp) * (in_b - b_zp)) * scale + out_zp.
 * 
 * For Saturn (a short-vector, decoupled microarchitecture), the bottleneck is often
 * instruction issue to the Vector Execute Sequencer (VXS). By specializing for 
 * a_zp == 0 and b_zp == 0, we reduce the VXS compute chain from 5 instructions 
 * (vwsub, vwsub, vwmul, vfcvt, vfmadd) to 3 instructions (vwmul, vfwcvt, vfmul); the output zero point is a saturating 16-bit add after rounding.
 * 
 * We use LMUL=2 for 8-bit types to achieve LMUL=8 for 32-bit intermediate floats,
 * maximizing chime length to hide functional unit latencies.
 */
void test_rvv(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int8_t a_zero_point = params->scalar.a_zero_point;
  const int8_t b_zero_point = params->scalar.b_zero_point;
  const float scale = params->scalar.scale;
  const int16_t ozp = (int16_t)params->scalar.output_zero_point;
  const int8_t output_min = params->scalar.output_min;
  const int8_t output_max = params->scalar.output_max;

  const size_t vlmax = __riscv_vsetvlmax_e8m2();

  size_t i = 0;
  size_t step = 2 * vlmax;

  // Zero-Point Specialization: Check if both zero points are zero to use a more efficient VXS compute chain.
  if (a_zero_point == 0 && b_zero_point == 0) {
    // Two full vectors per step while at least 2*vlmax elements remain.
    for (; i + step <= batch; i += step) {
      // Interleave memory loads to leverage VLS sequencer and enable arithmetic chaining.
      vint8m2_t va0 = __riscv_vle8_v_i8m2(input_a + i, vlmax);
      vint8m2_t vb0 = __riscv_vle8_v_i8m2(input_b + i, vlmax);
      vint8m2_t va1 = __riscv_vle8_v_i8m2(input_a + i + vlmax, vlmax);
      vint8m2_t vb1 = __riscv_vle8_v_i8m2(input_b + i + vlmax, vlmax);

      // 8x8 -> 16-bit widening multiply (1 VXS chime)
      vint16m4_t vacc16_0 = __riscv_vwmul_vv_i16m4(va0, vb0, vlmax);
      vint16m4_t vacc16_1 = __riscv_vwmul_vv_i16m4(va1, vb1, vlmax);

      // 16-bit int -> 32-bit float widening convert (1 VXS chime)
      vfloat32m8_t vfpacc0 = __riscv_vfwcvt_f_x_v_f32m8(vacc16_0, vlmax);
      vfloat32m8_t vfpacc1 = __riscv_vfwcvt_f_x_v_f32m8(vacc16_1, vlmax);

      // Scale (rounded once, as Neon); the zero point is added after rounding to int16
      vfpacc0 = __riscv_vfmul_vf_f32m8(vfpacc0, scale, vlmax);
      vfpacc1 = __riscv_vfmul_vf_f32m8(vfpacc1, scale, vlmax);

      // Narrowing Float-to-Int conversion
      vint16m4_t vout16_0 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc0, vlmax), ozp, vlmax);
      vint16m4_t vout16_1 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc1, vlmax), ozp, vlmax);

      // Narrow Int16 down to Int8 using saturating clip
      vint8m2_t vout0 = __riscv_vnclip_wx_i8m2(vout16_0, 0, __RISCV_VXRM_RDN, vlmax);
      vint8m2_t vout1 = __riscv_vnclip_wx_i8m2(vout16_1, 0, __RISCV_VXRM_RDN, vlmax);

      // Range clamping
      vout0 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout0, output_max, vlmax), output_min, vlmax);
      vout1 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout1, output_max, vlmax), output_min, vlmax);

      size_t remaining = batch - i;
      size_t vl0 = (remaining > vlmax) ? vlmax : remaining;
      __riscv_vse8_v_i8m2(output + i, vout0, vl0);
      if (remaining > vlmax) {
        size_t vl1 = remaining - vlmax;
        __riscv_vse8_v_i8m2(output + i + vlmax, vout1, vl1);
      }
    }
    // Tail: one vector at a time with vl = min(remaining, vlmax), so no load reads past the inputs.
    for (; i < batch; ) {
      const size_t vl = (batch - i < vlmax) ? batch - i : vlmax;
      vint8m2_t va0 = __riscv_vle8_v_i8m2(input_a + i, vl);
      vint8m2_t vb0 = __riscv_vle8_v_i8m2(input_b + i, vl);
      vint16m4_t vacc16_0 = __riscv_vwmul_vv_i16m4(va0, vb0, vl);
      vfloat32m8_t vfpacc0 = __riscv_vfwcvt_f_x_v_f32m8(vacc16_0, vl);
      vfpacc0 = __riscv_vfmul_vf_f32m8(vfpacc0, scale, vl);
      vint16m4_t vout16_0 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc0, vl), ozp, vl);
      vint8m2_t vout0 = __riscv_vnclip_wx_i8m2(vout16_0, 0, __RISCV_VXRM_RDN, vl);
      vout0 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout0, output_max, vl), output_min, vl);
      __riscv_vse8_v_i8m2(output + i, vout0, vl);
      i += vl;
    }
  } else {
    // Generic path for non-zero zero points
    // Two full vectors per step while at least 2*vlmax elements remain.
    for (; i + step <= batch; i += step) {
      vint8m2_t va0 = __riscv_vle8_v_i8m2(input_a + i, vlmax);
      vint8m2_t vb0 = __riscv_vle8_v_i8m2(input_b + i, vlmax);
      vint8m2_t va1 = __riscv_vle8_v_i8m2(input_a + i + vlmax, vlmax);
      vint8m2_t vb1 = __riscv_vle8_v_i8m2(input_b + i + vlmax, vlmax);

      // 8-bit to 16-bit widening subtract
      vint16m4_t vxa0 = __riscv_vwsub_vx_i16m4(va0, a_zero_point, vlmax);
      vint16m4_t vxb0 = __riscv_vwsub_vx_i16m4(vb0, b_zero_point, vlmax);
      vint16m4_t vxa1 = __riscv_vwsub_vx_i16m4(va1, a_zero_point, vlmax);
      vint16m4_t vxb1 = __riscv_vwsub_vx_i16m4(vb1, b_zero_point, vlmax);

      // 16-bit to 32-bit widening multiply
      vint32m8_t vacc0 = __riscv_vwmul_vv_i32m8(vxa0, vxb0, vlmax);
      vint32m8_t vacc1 = __riscv_vwmul_vv_i32m8(vxa1, vxb1, vlmax);

      // 32-bit int to 32-bit float conversion
      vfloat32m8_t vfpacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vlmax);
      vfloat32m8_t vfpacc1 = __riscv_vfcvt_f_x_v_f32m8(vacc1, vlmax);

      vfpacc0 = __riscv_vfmul_vf_f32m8(vfpacc0, scale, vlmax);
      vfpacc1 = __riscv_vfmul_vf_f32m8(vfpacc1, scale, vlmax);

      vint16m4_t vout16_0 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc0, vlmax), ozp, vlmax);
      vint16m4_t vout16_1 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc1, vlmax), ozp, vlmax);

      vint8m2_t vout0 = __riscv_vnclip_wx_i8m2(vout16_0, 0, __RISCV_VXRM_RDN, vlmax);
      vint8m2_t vout1 = __riscv_vnclip_wx_i8m2(vout16_1, 0, __RISCV_VXRM_RDN, vlmax);

      vout0 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout0, output_max, vlmax), output_min, vlmax);
      vout1 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout1, output_max, vlmax), output_min, vlmax);

      size_t remaining = batch - i;
      size_t vl0 = (remaining > vlmax) ? vlmax : remaining;
      __riscv_vse8_v_i8m2(output + i, vout0, vl0);
      if (remaining > vlmax) {
        size_t vl1 = remaining - vlmax;
        __riscv_vse8_v_i8m2(output + i + vlmax, vout1, vl1);
      }
    }
    // Tail: one vector at a time with vl = min(remaining, vlmax), so no load reads past the inputs.
    for (; i < batch; ) {
      const size_t vl = (batch - i < vlmax) ? batch - i : vlmax;
      vint8m2_t va0 = __riscv_vle8_v_i8m2(input_a + i, vl);
      vint8m2_t vb0 = __riscv_vle8_v_i8m2(input_b + i, vl);
      vint16m4_t vxa0 = __riscv_vwsub_vx_i16m4(va0, a_zero_point, vl);
      vint16m4_t vxb0 = __riscv_vwsub_vx_i16m4(vb0, b_zero_point, vl);
      vint32m8_t vacc0 = __riscv_vwmul_vv_i32m8(vxa0, vxb0, vl);
      vfloat32m8_t vfpacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vl);
      vfpacc0 = __riscv_vfmul_vf_f32m8(vfpacc0, scale, vl);
      vint16m4_t vout16_0 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4(vfpacc0, vl), ozp, vl);
      vint8m2_t vout0 = __riscv_vnclip_wx_i8m2(vout16_0, 0, __RISCV_VXRM_RDN, vl);
      vout0 = __riscv_vmax_vx_i8m2(__riscv_vmin_vx_i8m2(vout0, output_max, vl), output_min, vl);
      __riscv_vse8_v_i8m2(output + i, vout0, vl);
      i += vl;
    }
  }
}

