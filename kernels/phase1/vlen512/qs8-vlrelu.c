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

  const int16_t input_zero_point = (int16_t) params->scalar.input_zero_point;
  const int16_t positive_multiplier = (int16_t) (-params->scalar.positive_multiplier);
  const int16_t negative_multiplier = (int16_t) (-params->scalar.negative_multiplier);
  const int16_t output_zero_point = (int16_t) params->scalar.output_zero_point;

  for (; batch > 0; ) {
    const size_t vl = __riscv_vsetvl_e8m2(batch);
    const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl);
    input += vl;

    const vint16m4_t vx16 = __riscv_vsext_vf2_i16m4(vx, vl);
    vint16m4_t vacc = __riscv_vrsub_vx_i16m4(vx16, input_zero_point, vl);
    const vbool4_t vmask = __riscv_vmslt_vx_i16m4_b4(vacc, 0, vl);
    vacc = __riscv_vsll_vx_i16m4(vacc, 7, vl);

    vint16m4_t vmultiplier = __riscv_vmv_v_x_i16m4(negative_multiplier, vl);
    vmultiplier = __riscv_vmerge_vxm_i16m4(vmultiplier, positive_multiplier, vmask, vl);

    // [Rule 26] Doubling saturating rounding multiply high:
    // sat16((2 * a * b + 0x8000) >> 16) == sat16((a * b + 0x4000) >> 15).
    // Using shift 15 avoids 32-bit signed overflow when a = b = -32768.
    const vint32m8_t vprod = __riscv_vwmul_vv_i32m8(vacc, vmultiplier, vl);
    vacc = __riscv_vnclip_wx_i16m4(vprod, 15, __RISCV_VXRM_RNU, vl);
    vacc = __riscv_vsadd_vx_i16m4(vacc, output_zero_point, vl);

    const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(output, vy, vl);
    output += vl;

    batch -= vl;
  }
}