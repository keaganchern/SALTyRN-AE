#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const struct xnn_qs8_add_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zp = (int32_t)params->scalar.a_zero_point;
  const int32_t b_zp = (int32_t)params->scalar.b_zero_point;
  const int32_t a_mult = params->scalar.a_multiplier;
  const int32_t b_mult = params->scalar.b_multiplier;
  const uint32_t shift = (uint32_t)params->scalar.shift;
  const int32_t output_zero_point = (int32_t)params->scalar.output_zero_point;

  // Pre-calculate bias: K = -(a_zp * a_mult + b_zp * b_mult)
  // We use 64-bit math for the intermediate product to avoid overflow before the bias fold.
  const int32_t bias = (int32_t)-((int64_t)a_zp * a_mult + (int64_t)b_zp * b_mult);
  // Fold the output_zero_point into the bias. 
  // Shifting the zero point up allows it to be added before the final narrowing shift.
  // That is exact only while the accumulator plus (zero point << shift) fits int32: the products are below 2^30 in
  // magnitude and the shifted zero point at most 2^29 for shift <= 22. A larger shift takes the loop below, which
  // adds the zero point after the narrowing shift instead.
  const int fold_zero_point = shift <= 22;
  const int32_t folded_bias = fold_zero_point ? bias + (int32_t)((uint32_t)output_zero_point << shift) : bias;

  const int8_t output_min = params->scalar.output_min;
  const int8_t output_max = params->scalar.output_max;

  // Hoist loop-invariant bias vector. 
  // Saturn's VXS sequencer handles the splat (vmv.v.x). 
  // Using LMUL=8 maximizes chime length, helping hide VXS pipeline latencies (e.g., 3-stage MulPipe).
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  vint32m8_t vbias = __riscv_vmv_v_x_i32m8(folded_bias, vlmax);

  if (!fold_zero_point) {
    const int16_t ozp = (int16_t)output_zero_point;
    for (size_t vl; batch > 0; batch -= vl, input_a += vl, input_b += vl, output += vl) {
      vl = __riscv_vsetvl_e8m2(batch);

      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint32m8_t va32 = __riscv_vsext_vf4_i32m8(va, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);

      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_mult, vbias, vl);
      vint32m8_t vb32 = __riscv_vsext_vf4_i32m8(vb, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_mult, vb32, vl);

      vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RNU, vl);
      vacc16 = __riscv_vsadd_vx_i16m4(vacc16, ozp, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RDN, vl);

      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

      __riscv_vse8_v_i8m2(output, vout, vl);
    }
    return;
  }

  // Strategy 17: Saturation Logic Simplification.
  // vnclip inherently saturates to the target SEW range. If the clamp bounds are the 
  // native limits of int8 [-128, 127], we eliminate explicit vmax/vmin instructions.
  if (output_min == -128 && output_max == 127) {
    for (size_t vl; batch > 0; batch -= vl, input_a += vl, input_b += vl, output += vl) {
      vl = __riscv_vsetvl_e8m2(batch);

      // Interleave VLS (Load) and VXS (Execute) to leverage Saturn's DAE chaining.
      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      
      // Start widening A immediately; this chains with the load of A.
      vint32m8_t va32 = __riscv_vsext_vf4_i32m8(va, vl);

      // Load B while the VXS sequencer is busy with widening.
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);

      // Perform MADD: vacc = va32 * a_mult + vbias
      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_mult, vbias, vl);

      // Widen B and accumulate: vacc += vb32 * b_mult
      vint32m8_t vb32 = __riscv_vsext_vf4_i32m8(vb, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_mult, vb32, vl);

      // Narrowing sequence: 32-bit -> 16-bit -> 8-bit.
      // Saturn's vnclip is a 2-stage pipe; LMUL=8/4/2 ensures high occupancy.
      vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RNU, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RDN, vl);

      // VSS Sequencer: Store results. Chains with the final VXS narrowing.
      __riscv_vse8_v_i8m2(output, vout, vl);
    }
  } else {
    // Fallback path for custom clamping bounds.
    for (size_t vl; batch > 0; batch -= vl, input_a += vl, input_b += vl, output += vl) {
      vl = __riscv_vsetvl_e8m2(batch);

      vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
      vint32m8_t va32 = __riscv_vsext_vf4_i32m8(va, vl);
      vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);

      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_mult, vbias, vl);
      vint32m8_t vb32 = __riscv_vsext_vf4_i32m8(vb, vl);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_mult, vb32, vl);

      vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RNU, vl);
      vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RDN, vl);
      
      // Explicit clamping on VXS sequencer.
      vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
      vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);

      __riscv_vse8_v_i8m2(output, vout, vl);
    }
  }
}
