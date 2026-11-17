void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  kc = round_up_po2(kc, 2);
  do {
    vint32m4_t vksum = __riscv_vle32_v_i32m4((const int32_t*) w, 16);
    w = (const int32_t*) w + 16;
    const int32_t vzp0 = quantization_params[0].zero_point;
    vint32m4_t vacc = __riscv_vmul_vx_i32m4(vksum, vzp0, 16);

    const int8_t* a0 = a;
    for (size_t k = 0; k < kc; k += 2) {
      const int16_t a_val0 = (int16_t) *a0++;
      const int16_t a_val1 = (int16_t) *a0++;

      vint8m1_t wb = __riscv_vle8_v_i8m1((const int8_t*) w, 16);
      w = (const int8_t*) w + 16;

      vint8m1_t wb0 = __riscv_vsll_vx_i8m1(wb, 4, 16);
      vint8m1_t wb1 = __riscv_vand_vx_i8m1(wb, (int8_t) 0xF0, 16);

      vint16m2_t w16_0 = __riscv_vsext_vf2_i16m2(wb0, 16);
      vint16m2_t w16_1 = __riscv_vsext_vf2_i16m2(wb1, 16);

      vacc = __riscv_vwmacc_vx_i32m4(vacc, a_val0, w16_0, 16);
      vacc = __riscv_vwmacc_vx_i32m4(vacc, a_val1, w16_1, 16);
    }

    vint32m4_t vshifted = __riscv_vsra_vx_i32m4(vacc, 4, 16);
    vfloat32m4_t vout = __riscv_vfcvt_f_x_v_f32m4(vshifted, 16);

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout = __riscv_vfmul_vf_f32m4(vout, vinput_scale0, 16);

    vfloat32m4_t vfilter_output_scale = __riscv_vle32_v_f32m4((const float*) w, 16);
    w = (const float*) w + 16;

    vfloat32m4_t vbias = __riscv_vle32_v_f32m4((const float*) w, 16);
    w = (const float*) w + 16;

    vout = __riscv_vfmacc_vv_f32m4(vbias, vout, vfilter_output_scale, 16);

    const float voutput_min = params->scalar.min;
    vout = __riscv_vfmax_vf_f32m4(vout, voutput_min, 16);

    const float voutput_max = params->scalar.max;
    vout = __riscv_vfmin_vf_f32m4(vout, voutput_max, 16);

    if XNN_LIKELY(nc >= 16) {
      __riscv_vse32_v_f32m4(c0, vout, 16);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      nc -= 16;
    } else {
      __riscv_vse32_v_f32m4(c0, vout, nc);
      nc = 0;
    }
  } while (nc != 0);
}