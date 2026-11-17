#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(x) (x)
#endif

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

void test_rvv(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const float vmin_val = params->scalar.min;
  const float vmax_val = params->scalar.max;

  const float w0 = weights[0];
  const float w1 = weights[1];
  const float w2 = weights[2];
  const float w3 = weights[3];
  const float w4 = weights[4];
  const float w5 = weights[5];
  const float w6 = weights[6];
  const float w7 = weights[7];
  const float w8 = weights[8];
  const float w9 = weights[9];

  const size_t width = input_width / sizeof(float);

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

  size_t output_height = input_height;
  do {
    const float* cur_i0 = i0;
    const float* cur_i1 = i1;
    const float* cur_i2 = i2;
    const float* cur_i3 = i3;
    const float* cur_i4 = i4;

    float* cur_o0 = o0;
    float* cur_o1 = o1;
    float* cur_o2 = o2;

    if XNN_UNPREDICTABLE(output_height < 2) {
      cur_i2 = zero;
      cur_o1 = cur_o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      cur_i3 = zero;
      cur_o2 = cur_o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      cur_i4 = zero;
    }

    float prev0 = 0.0f;
    float prev1 = 0.0f;
    float prev2 = 0.0f;
    float prev3 = 0.0f;
    float prev4 = 0.0f;

    size_t w = width;
    while (w > 0) {
      const size_t vl = __riscv_vsetvl_e32m2(w);

      const vfloat32m2_t vi0 = __riscv_vle32_v_f32m2(cur_i0, vl);
      const vfloat32m2_t vi1 = __riscv_vle32_v_f32m2(cur_i1, vl);
      const vfloat32m2_t vi2 = __riscv_vle32_v_f32m2(cur_i2, vl);
      const vfloat32m2_t vi3 = __riscv_vle32_v_f32m2(cur_i3, vl);
      const vfloat32m2_t vi4 = __riscv_vle32_v_f32m2(cur_i4, vl);

      const float next0 = (w > vl) ? cur_i0[vl] : 0.0f;
      const float next1 = (w > vl) ? cur_i1[vl] : 0.0f;
      const float next2 = (w > vl) ? cur_i2[vl] : 0.0f;
      const float next3 = (w > vl) ? cur_i3[vl] : 0.0f;
      const float next4 = (w > vl) ? cur_i4[vl] : 0.0f;

      const vfloat32m2_t vi0_left = __riscv_vfslide1up_vf_f32m2(vi0, prev0, vl);
      const vfloat32m2_t vi1_left = __riscv_vfslide1up_vf_f32m2(vi1, prev1, vl);
      const vfloat32m2_t vi2_left = __riscv_vfslide1up_vf_f32m2(vi2, prev2, vl);
      const vfloat32m2_t vi3_left = __riscv_vfslide1up_vf_f32m2(vi3, prev3, vl);
      const vfloat32m2_t vi4_left = __riscv_vfslide1up_vf_f32m2(vi4, prev4, vl);

      const vfloat32m2_t vi0_right = __riscv_vfslide1down_vf_f32m2(vi0, next0, vl);
      const vfloat32m2_t vi1_right = __riscv_vfslide1down_vf_f32m2(vi1, next1, vl);
      const vfloat32m2_t vi2_right = __riscv_vfslide1down_vf_f32m2(vi2, next2, vl);
      const vfloat32m2_t vi3_right = __riscv_vfslide1down_vf_f32m2(vi3, next3, vl);
      const vfloat32m2_t vi4_right = __riscv_vfslide1down_vf_f32m2(vi4, next4, vl);

      prev0 = cur_i0[vl - 1];
      prev1 = cur_i1[vl - 1];
      prev2 = cur_i2[vl - 1];
      prev3 = cur_i3[vl - 1];
      prev4 = cur_i4[vl - 1];

      cur_i0 += vl;
      cur_i1 += vl;
      cur_i2 += vl;
      cur_i3 += vl;
      cur_i4 += vl;

      vfloat32m2_t vo0p0 = __riscv_vfmv_v_f_f32m2(w0, vl);
      vfloat32m2_t vo1p0 = __riscv_vfmv_v_f_f32m2(w0, vl);
      vfloat32m2_t vo2p0 = __riscv_vfmv_v_f_f32m2(w0, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w2, vi0, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w2, vi1, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w2, vi2, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w5, vi1, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w5, vi2, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w5, vi3, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w8, vi2, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w8, vi3, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w8, vi4, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w1, vi0_left, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w1, vi1_left, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w1, vi2_left, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w4, vi1_left, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w4, vi2_left, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w4, vi3_left, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w7, vi2_left, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w7, vi3_left, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w7, vi4_left, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w3, vi0_right, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w3, vi1_right, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w3, vi2_right, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w6, vi1_right, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w6, vi2_right, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w6, vi3_right, vl);

      vo0p0 = __riscv_vfmacc_vf_f32m2(vo0p0, w9, vi2_right, vl);
      vo1p0 = __riscv_vfmacc_vf_f32m2(vo1p0, w9, vi3_right, vl);
      vo2p0 = __riscv_vfmacc_vf_f32m2(vo2p0, w9, vi4_right, vl);

      const vbool16_t nan0 = __riscv_vmfne_vv_f32m2_b16(vo0p0, vo0p0, vl);
      const vbool16_t nan1 = __riscv_vmfne_vv_f32m2_b16(vo1p0, vo1p0, vl);
      const vbool16_t nan2 = __riscv_vmfne_vv_f32m2_b16(vo2p0, vo2p0, vl);

      vfloat32m2_t vo0 = __riscv_vfmax_vf_f32m2(vo0p0, vmin_val, vl);
      vfloat32m2_t vo1 = __riscv_vfmax_vf_f32m2(vo1p0, vmin_val, vl);
      vfloat32m2_t vo2 = __riscv_vfmax_vf_f32m2(vo2p0, vmin_val, vl);

      vo0 = __riscv_vfmin_vf_f32m2(vo0, vmax_val, vl);
      vo1 = __riscv_vfmin_vf_f32m2(vo1, vmax_val, vl);
      vo2 = __riscv_vfmin_vf_f32m2(vo2, vmax_val, vl);

      vo0 = __riscv_vmerge_vvm_f32m2(vo0, vo0p0, nan0, vl);
      vo1 = __riscv_vmerge_vvm_f32m2(vo1, vo1p0, nan1, vl);
      vo2 = __riscv_vmerge_vvm_f32m2(vo2, vo2p0, nan2, vl);

      __riscv_vse32_v_f32m2(cur_o2, vo2, vl); cur_o2 += vl;
      __riscv_vse32_v_f32m2(cur_o1, vo1, vl); cur_o1 += vl;
      __riscv_vse32_v_f32m2(cur_o0, vo0, vl); cur_o0 += vl;

      w -= vl;
    }

    i0 = i3;
    i1 = i4;
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    o0 = (float*) ((uintptr_t) o0 + 3 * input_width);
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = output_height >= 3 ? output_height - 3 : 0;
  } while (output_height != 0);
}