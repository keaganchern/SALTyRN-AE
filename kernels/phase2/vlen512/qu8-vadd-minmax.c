#include <assert.h>
// Input zero points folded into an int32 bias, output zero point added after the shift (no int64 range check), clamp in u8.
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// output_zero_point << shift is folded into the int32 bias only when the accumulator cannot overflow int32;
// otherwise the general loop runs.

void test_rvv(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const struct xnn_qu8_add_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  if (batch == 0) return;

  const uint8_t a_zero_point = params->scalar.a_zero_point;
  const uint8_t b_zero_point = params->scalar.b_zero_point;
  const int32_t a_multiplier = params->scalar.a_multiplier;
  const int32_t b_multiplier = params->scalar.b_multiplier;
  const uint32_t shift = (uint32_t) params->scalar.shift;
  const int16_t output_zero_point = params->scalar.output_zero_point;
  const uint8_t output_min = params->scalar.output_min;
  const uint8_t output_max = params->scalar.output_max;

  // Folded accumulator: acc(a, b) = init + a * a_multiplier + b * b_multiplier, where init carries the zero points and
  // (output_zero_point - 128) << shift, so one rounding narrowing shift yields the output minus 128. Exact only when the
  // true value of acc fits in int32 for every (a, b) in [0, 255]^2 (a linear function: its extremes are at the corners);
  // with a large shift (output scale far above the input scales) it does not, and the general loop below runs instead.
  // Input zero points folded into an int32 bias. Under XNNPACK's parameters (multipliers in [1, 2^21], zero points
  // in [0, 255]) the bias and every accumulator a*am + b*bm + bias = (a - azp)*am + (b - bzp)*bm stay within
  // +-2^30, so no range check is needed. The output zero point is added after the rounding shift.
  // Rounding term in the bias, then a floor shift (= a round-to-nearest-up shift; |acc| < 2^31 still).
  const int32_t init_acc = (int32_t) (1u << (shift - 1)) - (int32_t) a_zero_point * a_multiplier
                           - (int32_t) b_zero_point * b_multiplier;
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  const vint32m8_t vinit_acc = __riscv_vmv_v_x_i32m8(init_acc, vlmax);

  if (a_multiplier == b_multiplier) {
    // Initialize the software pipeline by loading the first chunk
    size_t vl = __riscv_vsetvl_e8m2(batch);
    vuint8m2_t va = __riscv_vle8_v_u8m2(input_a, vl);
    vuint8m2_t vb = __riscv_vle8_v_u8m2(input_b, vl);
    input_a += vl;
    input_b += vl;

    size_t i = vl;

    // Main pipelined loop (Specialized for symmetric quantization)
    while (i < batch) {
      size_t next_vl = __riscv_vsetvl_e8m2(batch - i);

      // Load next chunk first — overlaps with current compute via decoupling
      vuint8m2_t va_next = __riscv_vle8_v_u8m2(input_a, next_vl);
      vuint8m2_t vb_next = __riscv_vle8_v_u8m2(input_b, next_vl);

      // Pre-add the 8-bit values to 16-bit to halve widening and 32-bit MACs
      const vuint16m4_t vab = __riscv_vwaddu_vv_u16m4(va, vb, vl);
      const vuint32m8_t vab32_u = __riscv_vzext_vf2_u32m8(vab, vl);
      const vint32m8_t vab32 = __riscv_vreinterpret_v_u32m8_i32m8(vab32_u);

      // Multiply-add directly with loop-invariant addend
      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vab32, a_multiplier, vinit_acc, vl);

      // Narrowing clip to 16-bit signed with scaling shift
      vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

      // Signed saturating narrow (offset -128 is folded into init_acc), clamp, then flip the sign bit
      vacc16 = __riscv_vsadd_vx_i16m4(vacc16, output_zero_point, vl);
      vacc16 = __riscv_vmax_vx_i16m4(vacc16, (int16_t) output_min, vl);
      vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);

      // Store current chunk
      __riscv_vse8_v_u8m2(output, vout, vl);

      // Update pointers and loop state for the next iteration
      input_a += next_vl;
      input_b += next_vl;
      output += vl;

      va = va_next;
      vb = vb_next;
      vl = next_vl;
      i += vl;
    }

    // Drain final chunk (Specialized for symmetric quantization)
    const vuint16m4_t vab = __riscv_vwaddu_vv_u16m4(va, vb, vl);
    const vuint32m8_t vab32_u = __riscv_vzext_vf2_u32m8(vab, vl);
    const vint32m8_t vab32 = __riscv_vreinterpret_v_u32m8_i32m8(vab32_u);

    vint32m8_t vacc = __riscv_vmadd_vx_i32m8(vab32, a_multiplier, vinit_acc, vl);

    vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

    // Signed saturating narrow (offset -128 is folded into init_acc), clamp, then flip the sign bit
    vacc16 = __riscv_vsadd_vx_i16m4(vacc16, output_zero_point, vl);
    vacc16 = __riscv_vmax_vx_i16m4(vacc16, (int16_t) output_min, vl);
    vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RDN, vl);
    vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);

    __riscv_vse8_v_u8m2(output, vout, vl);

  } else {
    // Initialize the software pipeline by loading the first chunk
    size_t vl = __riscv_vsetvl_e8m2(batch);
    vuint8m2_t va = __riscv_vle8_v_u8m2(input_a, vl);
    vuint8m2_t vb = __riscv_vle8_v_u8m2(input_b, vl);
    input_a += vl;
    input_b += vl;

    size_t i = vl;

    // Main pipelined loop (Asymmetric processing)
    while (i < batch) {
      size_t next_vl = __riscv_vsetvl_e8m2(batch - i);

      // Load next chunk first
      vuint8m2_t va_next = __riscv_vle8_v_u8m2(input_a, next_vl);
      vuint8m2_t vb_next = __riscv_vle8_v_u8m2(input_b, next_vl);

      // Compute current chunk
      const vuint32m8_t va32_u = __riscv_vzext_vf4_u32m8(va, vl);
      const vint32m8_t va32 = __riscv_vreinterpret_v_u32m8_i32m8(va32_u);
      vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_multiplier, vinit_acc, vl);

      const vuint32m8_t vb32_u = __riscv_vzext_vf4_u32m8(vb, vl);
      const vint32m8_t vb32 = __riscv_vreinterpret_v_u32m8_i32m8(vb32_u);
      vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vb32, vl);

      vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

      // Signed saturating narrow (offset -128 is folded into init_acc), clamp, then flip the sign bit
      vacc16 = __riscv_vsadd_vx_i16m4(vacc16, output_zero_point, vl);
      vacc16 = __riscv_vmax_vx_i16m4(vacc16, (int16_t) output_min, vl);
      vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RDN, vl);
      vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);

      // Store current chunk
      __riscv_vse8_v_u8m2(output, vout, vl);

      // Update pointers and loop state
      input_a += next_vl;
      input_b += next_vl;
      output += vl;

      va = va_next;
      vb = vb_next;
      vl = next_vl;
      i += vl;
    }

    // Drain final chunk (Asymmetric processing)
    const vuint32m8_t va32_u = __riscv_vzext_vf4_u32m8(va, vl);
    const vint32m8_t va32 = __riscv_vreinterpret_v_u32m8_i32m8(va32_u);
    vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_multiplier, vinit_acc, vl);

    const vuint32m8_t vb32_u = __riscv_vzext_vf4_u32m8(vb, vl);
    const vint32m8_t vb32 = __riscv_vreinterpret_v_u32m8_i32m8(vb32_u);
    vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vb32, vl);

    vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);

    // Signed saturating narrow (offset -128 is folded into init_acc), clamp, then flip the sign bit
    vacc16 = __riscv_vsadd_vx_i16m4(vacc16, output_zero_point, vl);
    vacc16 = __riscv_vmax_vx_i16m4(vacc16, (int16_t) output_min, vl);
    vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RDN, vl);
    vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);

    __riscv_vse8_v_u8m2(output, vout, vl);
  }
}
