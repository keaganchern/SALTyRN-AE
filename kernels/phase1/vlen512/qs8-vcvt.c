#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t input_zero_point = params->scalar.input_zero_point;
  const int16_t multiplier = (int16_t)(-params->scalar.multiplier);
  const int16_t output_zero_point = params->scalar.output_zero_point;

  for (; batch > 0; ) {
    const size_t vl = __riscv_vsetvl_e8m2(batch);

    vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl);
    vint16m4_t vx_16 = __riscv_vsext_vf2_i16m4(vx, vl);

    // input_zero_point - vx
    vint16m4_t vacc = __riscv_vrsub_vx_i16m4(vx_16, input_zero_point, vl);
    vacc = __riscv_vsll_vx_i16m4(vacc, 7, vl);

    // vqrdmulhq_s16: widening multiply -> double -> vnclip with vxrm=RNU
    vint32m8_t vprod = __riscv_vwmul_vx_i32m8(vacc, multiplier, vl);
    vprod = __riscv_vsll_vx_i32m8(vprod, 1, vl);
    vacc = __riscv_vnclip_wx_i16m4(vprod, 16, __RISCV_VXRM_RNU, vl);

    // vqaddq_s16: saturating add
    vacc = __riscv_vsadd_vx_i16m4(vacc, output_zero_point, vl);

    // vqmovn_s16: saturating narrow to int8
    vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m2(output, vy, vl);

    input += vl;
    output += vl;
    batch -= vl;
  }
}