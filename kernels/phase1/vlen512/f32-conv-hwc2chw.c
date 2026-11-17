/*
struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};
*/

void test_rvv(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t vl4 = __riscv_vsetvl_e32m1(4);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float vmin_val = params->scalar.min;
  const float vmax_val = params->scalar.max;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o1c0 = output1;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      float vi0x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float vi1x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float vi2x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float vi3x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float vi4x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        vfloat32m1_t vo0x0 = __riscv_vle32_v_f32m1(w, vl4);
        vfloat32m1_t vo1x0 = vo0x0;
        vfloat32m1_t vo0x1 = vo0x0;
        vfloat32m1_t vo1x1 = vo0x0;

        const vfloat32m1_t vk00c0 = __riscv_vle32_v_f32m1(w + 4, vl4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        float vi0x1[4] = {i0[0], i0[1], i0[2], i0[3]}; i0 += 4;
        float vi1x1[4] = {i1[0], i1[1], i1[2], i1[3]}; i1 += 4;
        float vi2x1[4] = {i2[0], i2[1], i2[2], i2[3]}; i2 += 4;
        float vi3x1[4] = {i3[0], i3[1], i3[2], i3[3]}; i3 += 4;
        float vi4x1[4] = {i4[0], i4[1], i4[2], i4[3]}; i4 += 4;

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[1], vk00c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[1], vk00c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x1[3], vk00c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x1[3], vk00c0, vl4);

        const vfloat32m1_t vk10c0 = __riscv_vle32_v_f32m1(w + 8, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[1], vk10c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[1], vk10c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x1[3], vk10c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x1[3], vk10c0, vl4);

        const vfloat32m1_t vk20c0 = __riscv_vle32_v_f32m1(w + 12, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[1], vk20c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[1], vk20c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x1[3], vk20c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x1[3], vk20c0, vl4);

        const vfloat32m1_t vk00c1 = __riscv_vle32_v_f32m1(w + 16, vl4);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        float vi0x2[4] = {i0[0], i0[1], i0[2], i0[3]}; i0 += 4;
        float vi1x2[4] = {i1[0], i1[1], i1[2], i1[3]}; i1 += 4;
        float vi2x2[4] = {i2[0], i2[1], i2[2], i2[3]}; i2 += 4;
        float vi3x2[4] = {i3[0], i3[1], i3[2], i3[3]}; i3 += 4;
        float vi4x2[4] = {i4[0], i4[1], i4[2], i4[3]}; i4 += 4;

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[2], vk00c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[2], vk00c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[0], vk00c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[0], vk00c1, vl4);

        const vfloat32m1_t vk10c1 = __riscv_vle32_v_f32m1(w + 20, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[2], vk10c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[2], vk10c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[0], vk10c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[0], vk10c1, vl4);

        const vfloat32m1_t vk20c1 = __riscv_vle32_v_f32m1(w + 24, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[2], vk20c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[2], vk20c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[0], vk20c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[0], vk20c1, vl4);

        const vfloat32m1_t vk00c2 = __riscv_vle32_v_f32m1(w + 28, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[3], vk00c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[3], vk00c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[1], vk00c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[1], vk00c2, vl4);

        const vfloat32m1_t vk10c2 = __riscv_vle32_v_f32m1(w + 32, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[3], vk10c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[3], vk10c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[1], vk10c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[1], vk10c2, vl4);

        const vfloat32m1_t vk20c2 = __riscv_vle32_v_f32m1(w + 36, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[3], vk20c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[3], vk20c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[1], vk20c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[1], vk20c2, vl4);

        const vfloat32m1_t vk01c0 = __riscv_vle32_v_f32m1(w + 40, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[0], vk01c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[0], vk01c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[2], vk01c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[2], vk01c0, vl4);

        const vfloat32m1_t vk11c0 = __riscv_vle32_v_f32m1(w + 44, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[0], vk11c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[0], vk11c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[2], vk11c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[2], vk11c0, vl4);

        const vfloat32m1_t vk21c0 = __riscv_vle32_v_f32m1(w + 48, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[0], vk21c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[0], vk21c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[2], vk21c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[2], vk21c0, vl4);

        const vfloat32m1_t vk01c1 = __riscv_vle32_v_f32m1(w + 52, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[1], vk01c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[1], vk01c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[3], vk01c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[3], vk01c1, vl4);

        const vfloat32m1_t vk11c1 = __riscv_vle32_v_f32m1(w + 56, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[1], vk11c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[1], vk11c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[3], vk11c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[3], vk11c1, vl4);

        const vfloat32m1_t vk21c1 = __riscv_vle32_v_f32m1(w + 60, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[1], vk21c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[1], vk21c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[3], vk21c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[3], vk21c1, vl4);

        const vfloat32m1_t vk01c2 = __riscv_vle32_v_f32m1(w + 64, vl4);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        float vi0x3[4] = {i0[0], i0[1], i0[2], i0[3]}; i0 += 4;
        float vi1x3[4] = {i1[0], i1[1], i1[2], i1[3]}; i1 += 4;
        float vi2x3[4] = {i2[0], i2[1], i2[2], i2[3]}; i2 += 4;
        float vi3x3[4] = {i3[0], i3[1], i3[2], i3[3]}; i3 += 4;
        float vi4x3[4] = {i4[0], i4[1], i4[2], i4[3]}; i4 += 4;

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[2], vk01c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[2], vk01c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x3[0], vk01c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x3[0], vk01c2, vl4);

        const vfloat32m1_t vk11c2 = __riscv_vle32_v_f32m1(w + 68, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[2], vk11c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[2], vk11c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x3[0], vk11c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x3[0], vk11c2, vl4);

        const vfloat32m1_t vk21c2 = __riscv_vle32_v_f32m1(w + 72, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[2], vk21c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[2], vk21c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x3[0], vk21c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x3[0], vk21c2, vl4);

        const vfloat32m1_t vk02c0 = __riscv_vle32_v_f32m1(w + 76, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[3], vk02c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[3], vk02c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x3[1], vk02c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x3[1], vk02c0, vl4);

        const vfloat32m1_t vk12c0 = __riscv_vle32_v_f32m1(w + 80, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[3], vk12c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[3], vk12c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x3[1], vk12c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x3[1], vk12c0, vl4);

        const vfloat32m1_t vk22c0 = __riscv_vle32_v_f32m1(w + 84, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[3], vk22c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[3], vk22c0, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x3[1], vk22c0, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x3[1], vk22c0, vl4);

        const vfloat32m1_t vk02c1 = __riscv_vle32_v_f32m1(w + 88, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x2[0], vk02c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x2[0], vk02c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x3[2], vk02c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x3[2], vk02c1, vl4);

        const vfloat32m1_t vk12c1 = __riscv_vle32_v_f32m1(w + 92, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x2[0], vk12c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x2[0], vk12c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x3[2], vk12c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x3[2], vk12c1, vl4);

        const vfloat32m1_t vk22c1 = __riscv_vle32_v_f32m1(w + 96, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x2[0], vk22c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x2[0], vk22c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x3[2], vk22c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x3[2], vk22c1, vl4);

        const vfloat32m1_t vk02c2 = __riscv_vle32_v_f32m1(w + 100, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x2[1], vk02c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x2[1], vk02c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x3[3], vk02c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x3[3], vk02c2, vl4);

        const vfloat32m1_t vk12c2 = __riscv_vle32_v_f32m1(w + 104, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x2[1], vk12c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x2[1], vk12c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x3[3], vk12c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x3[3], vk12c2, vl4);

        const vfloat32m1_t vk22c2 = __riscv_vle32_v_f32m1(w + 108, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x2[1], vk22c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x2[1], vk22c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x3[3], vk22c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x3[3], vk22c2, vl4);

        vi0x0[0] = vi0x3[0]; vi0x0[1] = vi0x3[1]; vi0x0[2] = vi0x3[2]; vi0x0[3] = vi0x3[3];
        vi1x0[0] = vi1x3[0]; vi1x0[1] = vi1x3[1]; vi1x0[2] = vi1x3[2]; vi1x0[3] = vi1x3[3];
        vi2x0[0] = vi2x3[0]; vi2x0[1] = vi2x3[1]; vi2x0[2] = vi2x3[2]; vi2x0[3] = vi2x3[3];
        vi3x0[0] = vi3x3[0]; vi3x0[1] = vi3x3[1]; vi3x0[2] = vi3x3[2]; vi3x0[3] = vi3x3[3];
        vi4x0[0] = vi4x3[0]; vi4x0[1] = vi4x3[1]; vi4x0[2] = vi4x3[2]; vi4x0[3] = vi4x3[3];

        vo0x0 = __riscv_vfmax_vf_f32m1(vo0x0, vmin_val, vl4);
        vo1x0 = __riscv_vfmax_vf_f32m1(vo1x0, vmin_val, vl4);
        vo0x1 = __riscv_vfmax_vf_f32m1(vo0x1, vmin_val, vl4);
        vo1x1 = __riscv_vfmax_vf_f32m1(vo1x1, vmin_val, vl4);

        vo0x0 = __riscv_vfmin_vf_f32m1(vo0x0, vmax_val, vl4);
        vo1x0 = __riscv_vfmin_vf_f32m1(vo1x0, vmax_val, vl4);
        vo0x1 = __riscv_vfmin_vf_f32m1(vo0x1, vmax_val, vl4);
        vo1x1 = __riscv_vfmin_vf_f32m1(vo1x1, vmax_val, vl4);

        float out1_p0[4], out1_p1[4];
        float out0_p0[4], out0_p1[4];
        __riscv_vse32_v_f32m1(out1_p0, vo1x0, vl4);
        __riscv_vse32_v_f32m1(out1_p1, vo1x1, vl4);
        __riscv_vse32_v_f32m1(out0_p0, vo0x0, vl4);
        __riscv_vse32_v_f32m1(out0_p1, vo0x1, vl4);

        // Always 2+ output width elements remaining
        o1c0[0] = out1_p0[0]; o1c0[1] = out1_p1[0]; o1c0 += 2;
        o1c1[0] = out1_p0[1]; o1c1[1] = out1_p1[1]; o1c1 += 2;
        o1c2[0] = out1_p0[2]; o1c2[1] = out1_p1[2]; o1c2 += 2;
        o1c3[0] = out1_p0[3]; o1c3[1] = out1_p1[3]; o1c3 += 2;

        o0c0[0] = out0_p0[0]; o0c0[1] = out0_p1[0]; o0c0 += 2;
        o0c1[0] = out0_p0[1]; o0c1[1] = out0_p1[1]; o0c1 += 2;
        o0c2[0] = out0_p0[2]; o0c2[1] = out0_p1[2]; o0c2 += 2;
        o0c3[0] = out0_p0[3]; o0c3[1] = out0_p1[3]; o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        vfloat32m1_t vo0x0 = __riscv_vle32_v_f32m1(w, vl4);
        vfloat32m1_t vo1x0 = vo0x0;
        vfloat32m1_t vo0x1 = vo0x0;
        vfloat32m1_t vo1x1 = vo0x0;

        const vfloat32m1_t vk00c0 = __riscv_vle32_v_f32m1(w + 4, vl4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        float vi0x1[4] = {i0[0], i0[1], i0[2], i0[3]};
        float vi1x1[4] = {i1[0], i1[1], i1[2], i1[3]};
        float vi2x1[4] = {i2[0], i2[1], i2[2], i2[3]};
        float vi3x1[4] = {i3[0], i3[1], i3[2], i3[3]};
        float vi4x1[4] = {i4[0], i4[1], i4[2], i4[3]};

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[1], vk00c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[1], vk00c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x1[3], vk00c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x1[3], vk00c0, vl4);
        }

        const vfloat32m1_t vk10c0 = __riscv_vle32_v_f32m1(w + 8, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[1], vk10c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[1], vk10c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x1[3], vk10c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x1[3], vk10c0, vl4);
        }

        const vfloat32m1_t vk20c0 = __riscv_vle32_v_f32m1(w + 12, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[1], vk20c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[1], vk20c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x1[3], vk20c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x1[3], vk20c0, vl4);
        }

        const vfloat32m1_t vk00c1 = __riscv_vle32_v_f32m1(w + 16, vl4);

        float vi0x2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi1x2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi2x2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi3x2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi4x2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2[0] = i0[4]; vi0x2[1] = i0[5]; vi0x2[2] = i0[6]; vi0x2[3] = i0[7];
          vi1x2[0] = i1[4]; vi1x2[1] = i1[5]; vi1x2[2] = i1[6]; vi1x2[3] = i1[7];
          vi2x2[0] = i2[4]; vi2x2[1] = i2[5]; vi2x2[2] = i2[6]; vi2x2[3] = i2[7];
          vi3x2[0] = i3[4]; vi3x2[1] = i3[5]; vi3x2[2] = i3[6]; vi3x2[3] = i3[7];
          vi4x2[0] = i4[4]; vi4x2[1] = i4[5]; vi4x2[2] = i4[6]; vi4x2[3] = i4[7];
        }

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[2], vk00c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[2], vk00c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[0], vk00c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[0], vk00c1, vl4);

        const vfloat32m1_t vk10c1 = __riscv_vle32_v_f32m1(w + 20, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[2], vk10c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[2], vk10c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[0], vk10c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[0], vk10c1, vl4);

        const vfloat32m1_t vk20c1 = __riscv_vle32_v_f32m1(w + 24, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[2], vk20c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[2], vk20c1, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[0], vk20c1, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[0], vk20c1, vl4);

        const vfloat32m1_t vk00c2 = __riscv_vle32_v_f32m1(w + 28, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x0[3], vk00c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x0[3], vk00c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[1], vk00c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[1], vk00c2, vl4);

        const vfloat32m1_t vk10c2 = __riscv_vle32_v_f32m1(w + 32, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x0[3], vk10c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x0[3], vk10c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[1], vk10c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[1], vk10c2, vl4);

        const vfloat32m1_t vk20c2 = __riscv_vle32_v_f32m1(w + 36, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x0[3], vk20c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x0[3], vk20c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[1], vk20c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[1], vk20c2, vl4);

        const vfloat32m1_t vk01c0 = __riscv_vle32_v_f32m1(w + 40, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[0], vk01c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[0], vk01c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[2], vk01c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[2], vk01c0, vl4);
        }

        const vfloat32m1_t vk11c0 = __riscv_vle32_v_f32m1(w + 44, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[0], vk11c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[0], vk11c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[2], vk11c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[2], vk11c0, vl4);
        }

        const vfloat32m1_t vk21c0 = __riscv_vle32_v_f32m1(w + 48, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[0], vk21c0, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[0], vk21c0, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[2], vk21c0, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[2], vk21c0, vl4);
        }

        const vfloat32m1_t vk01c1 = __riscv_vle32_v_f32m1(w + 52, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[1], vk01c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[1], vk01c1, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x2[3], vk01c1, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x2[3], vk01c1, vl4);
        }

        const vfloat32m1_t vk11c1 = __riscv_vle32_v_f32m1(w + 56, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[1], vk11c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[1], vk11c1, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x2[3], vk11c1, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x2[3], vk11c1, vl4);
        }

        const vfloat32m1_t vk21c1 = __riscv_vle32_v_f32m1(w + 60, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[1], vk21c1, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[1], vk21c1, vl4);
        if (iw > 2) {
          vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x2[3], vk21c1, vl4);
          vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x2[3], vk21c1, vl4);
        }

        const vfloat32m1_t vk01c2 = __riscv_vle32_v_f32m1(w + 64, vl4);

        float vi0x3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi1x3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi2x3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi3x3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vi4x3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3[0] = *(i0 + 8);
          vi1x3[0] = *(i1 + 8);
          vi2x3[0] = *(i2 + 8);
          vi3x3[0] = *(i3 + 8);
          vi4x3[0] = *(i4 + 8);
        }

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[2], vk01c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[2], vk01c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi0x3[0], vk01c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi2x3[0], vk01c2, vl4);

        const vfloat32m1_t vk11c2 = __riscv_vle32_v_f32m1(w + 68, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[2], vk11c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[2], vk11c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi1x3[0], vk11c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi3x3[0], vk11c2, vl4);

        const vfloat32m1_t vk21c2 = __riscv_vle32_v_f32m1(w + 72, vl4);

        vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[2], vk21c2, vl4);
        vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[2], vk21c2, vl4);
        vo0x1 = __riscv_vfmacc_vf_f32m1(vo0x1, vi2x3[0], vk21c2, vl4);
        vo1x1 = __riscv_vfmacc_vf_f32m1(vo1x1, vi4x3[0], vk21c2, vl4);

        if (iw >= 2) {
          const vfloat32m1_t vk02c0 = __riscv_vle32_v_f32m1(w + 76, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x1[3], vk02c0, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x1[3], vk02c0, vl4);

          const vfloat32m1_t vk12c0 = __riscv_vle32_v_f32m1(w + 80, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x1[3], vk12c0, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x1[3], vk12c0, vl4);

          const vfloat32m1_t vk22c0 = __riscv_vle32_v_f32m1(w + 84, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x1[3], vk22c0, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x1[3], vk22c0, vl4);

          const vfloat32m1_t vk02c1 = __riscv_vle32_v_f32m1(w + 88, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x2[0], vk02c1, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x2[0], vk02c1, vl4);

          const vfloat32m1_t vk12c1 = __riscv_vle32_v_f32m1(w + 92, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x2[0], vk12c1, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x2[0], vk12c1, vl4);

          const vfloat32m1_t vk22c1 = __riscv_vle32_v_f32m1(w + 96, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x2[0], vk22c1, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x2[0], vk22c1, vl4);

          const vfloat32m1_t vk02c2 = __riscv_vle32_v_f32m1(w + 100, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi0x2[1], vk02c2, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi2x2[1], vk02c2, vl4);

          const vfloat32m1_t vk12c2 = __riscv_vle32_v_f32m1(w + 104, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi1x2[1], vk12c2, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi3x2[1], vk12c2, vl4);

          const vfloat32m1_t vk22c2 = __riscv_vle32_v_f32m1(w + 108, vl4);

          vo0x0 = __riscv_vfmacc_vf_f32m1(vo0x0, vi2x2[1], vk22c2, vl4);
          vo1x0 = __riscv_vfmacc_vf_f32m1(vo1x0, vi4x2[1], vk22c2, vl4);
        }

        vo0x0 = __riscv_vfmax_vf_f32m1(vo0x0, vmin_val, vl4);
        vo1x0 = __riscv_vfmax_vf_f32m1(vo1x0, vmin_val, vl4);
        vo0x1 = __riscv_vfmax_vf_f32m1(vo0x1, vmin_val, vl4);
        vo1x1 = __riscv_vfmax_vf_f32m1(vo1x1, vmin_val, vl4);

        vo0x0 = __riscv_vfmin_vf_f32m1(vo0x0, vmax_val, vl4);
        vo1x0 = __riscv_vfmin_vf_f32m1(vo1x0, vmax_val, vl4);
        vo0x1 = __riscv_vfmin_vf_f32m1(vo0x1, vmax_val, vl4);
        vo1x1 = __riscv_vfmin_vf_f32m1(vo1x1, vmax_val, vl4);

        if (iw == 3) {
          float out1_p0[4], out1_p1[4];
          float out0_p0[4], out0_p1[4];
          __riscv_vse32_v_f32m1(out1_p0, vo1x0, vl4);
          __riscv_vse32_v_f32m1(out1_p1, vo1x1, vl4);
          __riscv_vse32_v_f32m1(out0_p0, vo0x0, vl4);
          __riscv_vse32_v_f32m1(out0_p1, vo0x1, vl4);

          // Exactly 2 output width elements remaining
          o1c0[0] = out1_p0[0]; o1c0[1] = out1_p1[0]; o1c0 += 2;
          o1c1[0] = out1_p0[1]; o1c1[1] = out1_p1[1]; o1c1 += 2;
          o1c2[0] = out1_p0[2]; o1c2[1] = out1_p1[2]; o1c2 += 2;
          o1c3[0] = out1_p0[3]; o1c3[1] = out1_p1[3]; o1c3 += 2;

          o0c0[0] = out0_p0[0]; o0c0[1] = out0_p1[0]; o0c0 += 2;
          o0c1[0] = out0_p0[1]; o0c1[1] = out0_p1[1]; o0c1 += 2;
          o0c2[0] = out0_p0[2]; o0c2[1] = out0_p1[2]; o0c2 += 2;
          o0c3[0] = out0_p0[3]; o0c3[1] = out0_p1[3]; o0c3 += 2;
        } else {
          float out1_p0[4];
          float out0_p0[4];
          __riscv_vse32_v_f32m1(out1_p0, vo1x0, vl4);
          __riscv_vse32_v_f32m1(out0_p0, vo0x0, vl4);

          // Exactly 1 output width element remaining
          o1c0[0] = out1_p0[0]; o1c0 += 1;
          o1c1[0] = out1_p0[1]; o1c1 += 1;
          o1c2[0] = out1_p0[2]; o1c2 += 1;
          o1c3[0] = out1_p0[3]; o1c3 += 1;

          o0c0[0] = out0_p0[0]; o0c0 += 1;
          o0c1[0] = out0_p0[1]; o0c1 += 1;
          o0c2[0] = out0_p0[2]; o0c2 += 1;
          o0c3[0] = out0_p0[3]; o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (float*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (float*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (float*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}