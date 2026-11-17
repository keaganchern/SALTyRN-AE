#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

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

  const float params_min = params->scalar.min;
  const float params_max = params->scalar.max;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    // Determine row pointers for the current 2x2 output row block
    intptr_t iy0 = (intptr_t)output_y * 2 - (intptr_t)input_padding_top;
    const float* i_ptrs[5];
    for (int i = 0; i < 5; ++i) {
        intptr_t iy = iy0 + i;
        if (iy < 0 || iy >= (intptr_t)input_height) {
            i_ptrs[i] = zero;
        } else {
            i_ptrs[i] = (const float*)((uintptr_t)input + (size_t)iy * input_height_stride);
        }
    }

    const float* w_base = weights;
    size_t c = output_channels;
    float* o0_chan_ptr = (float*)((uintptr_t)output + output_height_stride * output_y);
    float* o1_chan_ptr = (float*)((uintptr_t)o0_chan_ptr + output_height_stride);
    const bool has_next_y = (output_y + 1 < output_y_end);

    while (c > 0) {
      size_t vl = __riscv_vsetvl_e32m4(c);
      vfloat32m4_t v_bias = __riscv_vle32_v_f32m4(w_base, vl);
      float* o0_c = o0_chan_ptr;
      float* o1_c = o1_chan_ptr;

      vfloat32m4_t vo0_0 = v_bias;
      vfloat32m4_t vo1_0 = v_bias;
      vfloat32m4_t vo0_1 = v_bias;
      vfloat32m4_t vo1_1 = v_bias;

      for (size_t ox = 0; ox < output_width; ) {
        const float* w_ptr = w_base + vl;
        
        // Steady-state path with Dual-Issue and scalar optimization
        if ((ox + 1 < output_width) && ((intptr_t)ox * 2 - 1 >= 0) && ((intptr_t)ox * 2 + 3 < (intptr_t)input_width)) {
          vfloat32m4_t wy = __riscv_vle32_v_f32m4(w_ptr, vl); w_ptr += vl;
          vfloat32m4_t wy_next;

          // Precalculate base offsets to reduce scalar pressure in the tight inner loop
          intptr_t base_idx0 = (2 * (intptr_t)ox - 1) * 3;
          intptr_t base_idx1 = (2 * (intptr_t)(ox + 1) - 1) * 3;

          #define P_SUBSTEP(OFF, FY, IS_LAST) { \
              intptr_t idx0 = base_idx0 + (OFF); \
              intptr_t idx1 = base_idx1 + (OFF); \
              float s00 = i_ptrs[FY][idx0], s10 = i_ptrs[FY+2][idx0]; \
              float s01 = i_ptrs[FY][idx1], s11 = i_ptrs[FY+2][idx1]; \
              if (!(IS_LAST)) { wy_next = __riscv_vle32_v_f32m4(w_ptr, vl); w_ptr += vl; } \
              vo0_0 = __riscv_vfmacc_vf_f32m4(vo0_0, s00, wy, vl); \
              vo1_0 = __riscv_vfmacc_vf_f32m4(vo1_0, s10, wy, vl); \
              vo0_1 = __riscv_vfmacc_vf_f32m4(vo0_1, s01, wy, vl); \
              vo1_1 = __riscv_vfmacc_vf_f32m4(vo1_1, s11, wy, vl); \
              if (!(IS_LAST)) wy = wy_next; \
          }

          #define P_FX_FC(OFF) \
              P_SUBSTEP(OFF, 0, 0) \
              P_SUBSTEP(OFF, 1, 0) \
              P_SUBSTEP(OFF, 2, ((OFF) == 8))

          P_FX_FC(0) P_FX_FC(1) P_FX_FC(2)
          P_FX_FC(3) P_FX_FC(4) P_FX_FC(5)
          P_FX_FC(6) P_FX_FC(7) P_FX_FC(8)
          
          #undef P_SUBSTEP
          #undef P_FX_FC

          // Interleave IntPipe (move) and FPConvPipe (min/max) to maximize pipeline utilization
          vo0_0 = __riscv_vfmax_vf_f32m4(vo0_0, params_min, vl);
          vfloat32m4_t next_vo0_0 = v_bias;
          vo0_0 = __riscv_vfmin_vf_f32m4(vo0_0, params_max, vl);
          
          vo1_0 = __riscv_vfmax_vf_f32m4(vo1_0, params_min, vl);
          vfloat32m4_t next_vo1_0 = v_bias;
          vo1_0 = __riscv_vfmin_vf_f32m4(vo1_0, params_max, vl);
          
          vo0_1 = __riscv_vfmax_vf_f32m4(vo0_1, params_min, vl);
          vfloat32m4_t next_vo0_1 = v_bias;
          vo0_1 = __riscv_vfmin_vf_f32m4(vo0_1, params_max, vl);
          
          vo1_1 = __riscv_vfmax_vf_f32m4(vo1_1, params_min, vl);
          vfloat32m4_t next_vo1_1 = v_bias;
          vo1_1 = __riscv_vfmin_vf_f32m4(vo1_1, params_max, vl);
          
          __riscv_vssseg2e32_v_f32m4x2(o0_c + ox, output_channel_stride, __riscv_vcreate_v_f32m4x2(vo0_0, vo0_1), vl);
          if (has_next_y) {
            __riscv_vssseg2e32_v_f32m4x2(o1_c + ox, output_channel_stride, __riscv_vcreate_v_f32m4x2(vo1_0, vo1_1), vl);
          }
          
          ox += 2;
          vo0_0 = next_vo0_0; vo1_0 = next_vo1_0; vo0_1 = next_vo0_1; vo1_1 = next_vo1_1;
        } else {
          // Boundary/Padded Fallback Logic
          for (int fx = 0; fx < 3; ++fx) {
            for (int fc = 0; fc < 3; ++fc) {
              vfloat32m4_t wy0 = __riscv_vle32_v_f32m4(w_ptr, vl); w_ptr += vl;
              vfloat32m4_t wy1 = __riscv_vle32_v_f32m4(w_ptr, vl); w_ptr += vl;
              vfloat32m4_t wy2 = __riscv_vle32_v_f32m4(w_ptr, vl); w_ptr += vl;
              
              intptr_t ix0 = 2 * (intptr_t)ox + fx - 1;
              // As in Neon: the left padding column (ix0 = -1) is multiplied as 0 (0 * w is NaN for a non-finite
              // w), a column past the right edge (odd width) is skipped.
              if ((size_t)(ix0 + 1) <= input_width) {
                intptr_t idx0 = ix0 >= 0 ? ix0 * 3 + fc : 0;
                vo0_0 = __riscv_vfmacc_vf_f32m4(vo0_0, (ix0 >= 0 ? i_ptrs[0][idx0] : 0.0f), wy0, vl);
                vo0_0 = __riscv_vfmacc_vf_f32m4(vo0_0, (ix0 >= 0 ? i_ptrs[1][idx0] : 0.0f), wy1, vl);
                vo0_0 = __riscv_vfmacc_vf_f32m4(vo0_0, (ix0 >= 0 ? i_ptrs[2][idx0] : 0.0f), wy2, vl);
                vo1_0 = __riscv_vfmacc_vf_f32m4(vo1_0, (ix0 >= 0 ? i_ptrs[2][idx0] : 0.0f), wy0, vl);
                vo1_0 = __riscv_vfmacc_vf_f32m4(vo1_0, (ix0 >= 0 ? i_ptrs[3][idx0] : 0.0f), wy1, vl);
                vo1_0 = __riscv_vfmacc_vf_f32m4(vo1_0, (ix0 >= 0 ? i_ptrs[4][idx0] : 0.0f), wy2, vl);
              }
              if (ox + 1 < output_width) {
                intptr_t ix1 = 2 * (intptr_t)(ox + 1) + fx - 1;
                if (ix1 >= 0 && (size_t)ix1 < input_width) {
                  intptr_t idx1 = ix1 * 3 + fc;
                  vo0_1 = __riscv_vfmacc_vf_f32m4(vo0_1, i_ptrs[0][idx1], wy0, vl);
                  vo0_1 = __riscv_vfmacc_vf_f32m4(vo0_1, i_ptrs[1][idx1], wy1, vl);
                  vo0_1 = __riscv_vfmacc_vf_f32m4(vo0_1, i_ptrs[2][idx1], wy2, vl);
                  vo1_1 = __riscv_vfmacc_vf_f32m4(vo1_1, i_ptrs[2][idx1], wy0, vl);
                  vo1_1 = __riscv_vfmacc_vf_f32m4(vo1_1, i_ptrs[3][idx1], wy1, vl);
                  vo1_1 = __riscv_vfmacc_vf_f32m4(vo1_1, i_ptrs[4][idx1], wy2, vl);
                }
              }
            }
          }
          vo0_0 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vo0_0, params_min, vl), params_max, vl);
          vo1_0 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vo1_0, params_min, vl), params_max, vl);
          if (ox + 1 < output_width) {
            vo0_1 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vo0_1, params_min, vl), params_max, vl);
            vo1_1 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vo1_1, params_min, vl), params_max, vl);
            __riscv_vssseg2e32_v_f32m4x2(o0_c + ox, output_channel_stride, __riscv_vcreate_v_f32m4x2(vo0_0, vo0_1), vl);
            if (has_next_y) {
              __riscv_vssseg2e32_v_f32m4x2(o1_c + ox, output_channel_stride, __riscv_vcreate_v_f32m4x2(vo1_0, vo1_1), vl);
            }
          } else {
            __riscv_vsse32_v_f32m4(o0_c + ox, output_channel_stride, vo0_0, vl);
            if (has_next_y) {
              __riscv_vsse32_v_f32m4(o1_c + ox, output_channel_stride, vo1_0, vl);
            }
          }
          ox += 2;
          vo0_0 = v_bias; vo1_0 = v_bias; vo0_1 = v_bias; vo1_1 = v_bias;
        }
      }
      w_base += 28 * vl;
      o0_chan_ptr = (float*) ((uintptr_t) o0_chan_ptr + vl * output_channel_stride);
      o1_chan_ptr = (float*) ((uintptr_t) o1_chan_ptr + vl * output_channel_stride);
      c -= vl;
    }
  }
}

