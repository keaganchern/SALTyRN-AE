#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// Each output row sums its 27 taps in Neon's order (kernel column, input channel, row) from three unit-stride
// segment-loaded input rows; the middle input row is shared by the two output rows. The first block does not load
// lane 0's left pixel (padding, set to 0); an odd-width row's last pixel runs alone without its right column, as
// in Neon. The vector blocks in between run unmasked and branch-free.

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
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if (output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float min_f = params->scalar.min;
  const float max_f = params->scalar.max;


// One input row: pixels 2x-1 (R0..R2) and 2x (R3..R5) of every lane in one segment load. In the first block lane 0
// is not loaded; its left pixel is padding (0) and its pixel 0 is read from the row.
#define LOAD_ROW(R, i_ptr, FIRST) \
        if (FIRST) { v_px = __riscv_vlseg6e32_v_f32m1x6_m(mask0, i_ptr, vl); } \
        else { v_px = __riscv_vlseg6e32_v_f32m1x6(i_ptr, vl); } \
        R##0 = __riscv_vget_v_f32m1x6_f32m1(v_px, 0); \
        R##1 = __riscv_vget_v_f32m1x6_f32m1(v_px, 1); \
        R##2 = __riscv_vget_v_f32m1x6_f32m1(v_px, 2); \
        R##3 = __riscv_vget_v_f32m1x6_f32m1(v_px, 3); \
        R##4 = __riscv_vget_v_f32m1x6_f32m1(v_px, 4); \
        R##5 = __riscv_vget_v_f32m1x6_f32m1(v_px, 5); \
        if (FIRST) { \
          R##0 = __riscv_vfmv_s_f_f32m1_tu(R##0, 0.0f, vl); \
          R##1 = __riscv_vfmv_s_f_f32m1_tu(R##1, 0.0f, vl); \
          R##2 = __riscv_vfmv_s_f_f32m1_tu(R##2, 0.0f, vl); \
          R##3 = __riscv_vfmv_s_f_f32m1_tu(R##3, i_ptr[3], vl); \
          R##4 = __riscv_vfmv_s_f_f32m1_tu(R##4, i_ptr[4], vl); \
          R##5 = __riscv_vfmv_s_f_f32m1_tu(R##5, i_ptr[5], vl); \
        }

// acc[oc] += w[tap][oc] * v for the 4 output channels; tap = dx*9 + c_in*3 + ky (Neon's plane order).
#define MAC4(O, tap, v) \
        O##c0 = __riscv_vfmacc_vf_f32m1(O##c0, w[4 + (tap) * 4 + 0], v, vl); \
        O##c1 = __riscv_vfmacc_vf_f32m1(O##c1, w[4 + (tap) * 4 + 1], v, vl); \
        O##c2 = __riscv_vfmacc_vf_f32m1(O##c2, w[4 + (tap) * 4 + 2], v, vl); \
        O##c3 = __riscv_vfmacc_vf_f32m1(O##c3, w[4 + (tap) * 4 + 3], v, vl);

// Right pixel 2x+1, channel C: the next lane's pixel 2x-1; the last lane's comes from the row.
#define RIGHT(R, C, i_ptr) vt = __riscv_vfslide1down_vf_f32m1(R##C, i_ptr[vl * 6 + C], vl);

// One output row from input rows A, B, C (ky = 0, 1, 2): bias, then each kernel column dx and input channel
// c_in, rows top to bottom. NORIGHT drops the right column (dx = 2).
#define OUTPUT_ROW(O, A, B, C, pa, pb, pc, NORIGHT) \
        MAC4(O, 0, A##0) MAC4(O, 1, B##0) MAC4(O, 2, C##0) \
        MAC4(O, 3, A##1) MAC4(O, 4, B##1) MAC4(O, 5, C##1) \
        MAC4(O, 6, A##2) MAC4(O, 7, B##2) MAC4(O, 8, C##2) \
        MAC4(O, 9, A##3) MAC4(O, 10, B##3) MAC4(O, 11, C##3) \
        MAC4(O, 12, A##4) MAC4(O, 13, B##4) MAC4(O, 14, C##4) \
        MAC4(O, 15, A##5) MAC4(O, 16, B##5) MAC4(O, 17, C##5) \
        if (!(NORIGHT)) { \
          RIGHT(A, 0, pa) MAC4(O, 18, vt) RIGHT(B, 0, pb) MAC4(O, 19, vt) RIGHT(C, 0, pc) MAC4(O, 20, vt) \
          RIGHT(A, 1, pa) MAC4(O, 21, vt) RIGHT(B, 1, pb) MAC4(O, 22, vt) RIGHT(C, 1, pc) MAC4(O, 23, vt) \
          RIGHT(A, 2, pa) MAC4(O, 24, vt) RIGHT(B, 2, pb) MAC4(O, 25, vt) RIGHT(C, 2, pc) MAC4(O, 26, vt) \
        }

#define CLAMP4(O) \
        O##c0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(O##c0, min_f, vl), max_f, vl); \
        O##c1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(O##c1, min_f, vl), max_f, vl); \
        O##c2 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(O##c2, min_f, vl), max_f, vl); \
        O##c3 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(O##c3, min_f, vl), max_f, vl);

// One block of vl output columns, both output rows. Row 0 is stored before row 1 is computed (fewer live
// registers); when row 1 aliases row 0 (odd output height) it is not stored, so row 0's values stay.
#define BLOCK(FIRST, NORIGHT) \
      { \
        const float* w = w_base; \
        vbool32_t mask0 = __riscv_vmsne_vx_u32m1_b32(__riscv_vid_v_u32m1(vl), 0, vl); \
        vfloat32m1x6_t v_px; \
        vfloat32m1_t rA0, rA1, rA2, rA3, rA4, rA5, rB0, rB1, rB2, rB3, rB4, rB5, rC0, rC1, rC2, rC3, rC4, rC5, vt; \
        (void) mask0; (void) vt; \
        vfloat32m1_t v_o0c0 = __riscv_vfmv_v_f_f32m1(w[0], vl); \
        vfloat32m1_t v_o0c1 = __riscv_vfmv_v_f_f32m1(w[1], vl); \
        vfloat32m1_t v_o0c2 = __riscv_vfmv_v_f_f32m1(w[2], vl); \
        vfloat32m1_t v_o0c3 = __riscv_vfmv_v_f_f32m1(w[3], vl); \
        LOAD_ROW(rA, i0_ptr, FIRST) \
        LOAD_ROW(rB, i1_ptr, FIRST) \
        LOAD_ROW(rC, i2_ptr, FIRST) \
        OUTPUT_ROW(v_o0, rA, rB, rC, i0_ptr, i1_ptr, i2_ptr, NORIGHT) \
        CLAMP4(v_o0) \
        __riscv_vse32_v_f32m1(o0c0, v_o0c0, vl); \
        __riscv_vse32_v_f32m1(o0c1, v_o0c1, vl); \
        __riscv_vse32_v_f32m1(o0c2, v_o0c2, vl); \
        __riscv_vse32_v_f32m1(o0c3, v_o0c3, vl); \
        vfloat32m1_t v_o1c0 = __riscv_vfmv_v_f_f32m1(w[0], vl); \
        vfloat32m1_t v_o1c1 = __riscv_vfmv_v_f_f32m1(w[1], vl); \
        vfloat32m1_t v_o1c2 = __riscv_vfmv_v_f_f32m1(w[2], vl); \
        vfloat32m1_t v_o1c3 = __riscv_vfmv_v_f_f32m1(w[3], vl); \
        LOAD_ROW(rB, i3_ptr, FIRST) \
        LOAD_ROW(rA, i4_ptr, FIRST) \
        OUTPUT_ROW(v_o1, rC, rB, rA, i2_ptr, i3_ptr, i4_ptr, NORIGHT) \
        CLAMP4(v_o1) \
        if (o1c0 != o0c0) { \
          __riscv_vse32_v_f32m1(o1c0, v_o1c0, vl); \
          __riscv_vse32_v_f32m1(o1c1, v_o1c1, vl); \
          __riscv_vse32_v_f32m1(o1c2, v_o1c2, vl); \
          __riscv_vse32_v_f32m1(o1c3, v_o1c3, vl); \
        } \
        o0c0 += vl; o0c1 += vl; o0c2 += vl; o0c3 += vl; \
        o1c0 += vl; o1c1 += vl; o1c2 += vl; o1c3 += vl; \
        i0_ptr += vl * 6; i1_ptr += vl * 6; i2_ptr += vl * 6; i3_ptr += vl * 6; i4_ptr += vl * 6; \
      }

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if (input_y2 >= input_height) { i2 = zero; }
    if (input_y4 > input_height) { i3 = zero; }
    if (input_y4 >= input_height) { i4 = zero; }
    if (output_y + 2 > output_y_end) { output1 = output0; }

    const float* i0_base = (const float*)((uintptr_t)i0 - 3 * sizeof(float));
    const float* i1_base = (const float*)((uintptr_t)i1 - 3 * sizeof(float));
    const float* i2_base = (const float*)((uintptr_t)i2 - 3 * sizeof(float));
    const float* i3_base = (const float*)((uintptr_t)i3 - 3 * sizeof(float));
    const float* i4_base = (const float*)((uintptr_t)i4 - 3 * sizeof(float));

    const float* w_base = weights;
    size_t c = output_channels;
    float* current_o0 = output0;
    float* current_o1 = output1;
    // Output columns whose right pixel 2x+1 is inside the row: all of them for an even width, all but the last
    // for an odd width.
    const size_t ow_vec = (input_width & 1) ? output_width - 1 : output_width;

    do {
      float* o0c0 = current_o0;
      float* o1c0 = current_o1;
      float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
      float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
      float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
      float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
      float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
      float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);

      if (c < 2) { o0c1 = o0c0; o1c1 = o1c0; }
      if (c <= 2) { o0c2 = o0c1; o1c2 = o1c1; }
      if (c < 4) { o0c3 = o0c2; o1c3 = o1c2; }

      const float* i0_ptr = i0_base;
      const float* i1_ptr = i1_base;
      const float* i2_ptr = i2_base;
      const float* i3_ptr = i3_base;
      const float* i4_ptr = i4_base;
      size_t rem = ow_vec;
      if (rem > 0) {
        size_t vl = __riscv_vsetvl_e32m1(rem);
        BLOCK(1, 0)
        rem -= vl;
      }
      while (rem > 0) {
        size_t vl = __riscv_vsetvl_e32m1(rem);
        BLOCK(0, 0)
        rem -= vl;
      }
      if (ow_vec < output_width) {      // odd width: the last pixel, without its right column
        size_t vl = __riscv_vsetvl_e32m1(1);
        if (ow_vec == 0) { BLOCK(1, 1) } else { BLOCK(0, 1) }
      }

      current_o0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      current_o1 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      w_base += 112;
      c = (c > 4) ? c - 4 : 0;
    } while (c != 0);

    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
  }
#undef BLOCK
#undef CLAMP4
#undef OUTPUT_ROW
#undef RIGHT
#undef MAC4
#undef LOAD_ROW
}
