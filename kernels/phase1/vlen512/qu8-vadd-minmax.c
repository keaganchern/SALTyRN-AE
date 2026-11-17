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

  const uint8_t a_zero_point = params->scalar.a_zero_point;
  const uint8_t b_zero_point = params->scalar.b_zero_point;
  const int32_t a_multiplier = params->scalar.a_multiplier;
  const int32_t b_multiplier = params->scalar.b_multiplier;
  const uint32_t shift = (uint32_t) params->scalar.shift;
  const int16_t output_zero_point = params->scalar.output_zero_point;
  const uint8_t output_min = params->scalar.output_min;
  const uint8_t output_max = params->scalar.output_max;

  for (; batch > 0; ) {
    const size_t vl = __riscv_vsetvl_e8m2(batch);

    const vuint8m2_t va = __riscv_vle8_v_u8m2(input_a, vl);
    const vuint8m2_t vb = __riscv_vle8_v_u8m2(input_b, vl);

    const vuint16m4_t vdiff_a = __riscv_vwsubu_vx_u16m4(va, a_zero_point, vl);
    const vuint16m4_t vdiff_b = __riscv_vwsubu_vx_u16m4(vb, b_zero_point, vl);

    const vint16m4_t vxa = __riscv_vreinterpret_v_u16m4_i16m4(vdiff_a);
    const vint16m4_t vxb = __riscv_vreinterpret_v_u16m4_i16m4(vdiff_b);

    const vint32m8_t vxa32 = __riscv_vsext_vf2_i32m8(vxa, vl);
    const vint32m8_t vxb32 = __riscv_vsext_vf2_i32m8(vxb, vl);

    vint32m8_t vacc = __riscv_vmul_vx_i32m8(vxa32, a_multiplier, vl);
    vacc = __riscv_vmacc_vx_i32m8(vacc, b_multiplier, vxb32, vl);

    vacc = __riscv_vssra_vx_i32m8(vacc, shift, __RISCV_VXRM_RNU, vl);

    vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, 0, __RISCV_VXRM_RNU, vl);
    vacc16 = __riscv_vsadd_vx_i16m4(vacc16, output_zero_point, vl);

    const vint16m4_t vacc16_clamped = __riscv_vmax_vx_i16m4(vacc16, 0, vl);
    const vuint16m4_t vacc16_u = __riscv_vreinterpret_v_i16m4_u16m4(vacc16_clamped);
    vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(vacc16_u, 0, __RISCV_VXRM_RNU, vl);

    vout = __riscv_vmaxu_vx_u8m2(vout, output_min, vl);
    vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);

    __riscv_vse8_v_u8m2(output, vout, vl);

    input_a += vl;
    input_b += vl;
    output += vl;
    batch -= vl;
  }
}