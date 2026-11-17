#include <riscv_vector.h>
#include "xnnpack/math.h"

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
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float min_f = params->scalar.min;
  const float max_f = params->scalar.max;
  const int32_t vzp0 = quantization_params[0].zero_point;
  const float inv_scale = quantization_params[0].inv_scale;

  kc = round_up_po2(kc, 2);

  while (nc > 0) {
    // The weights are packed in blocks of 16 columns.
    size_t vl = __riscv_vsetvl_e32m8(nc > 16 ? 16 : nc);

    // Initialize accumulators. 
    // vacc0 is initialized with the scaled base weights from the packed buffer.
    vint32m8_t vacc0 = __riscv_vle32_v_i32m8((const int32_t*)w, vl);
    vacc0 = __riscv_vmul_vx_i32m8(vacc0, vzp0, vl);
    
    // vacc1 is initialized to zero to break RAW dependency chains on vacc0.
    vint32m8_t vacc1 = __riscv_vmv_v_x_i32m8(0, vl);

    const int8_t* w_weights = (const int8_t*)w + 64;
    const int8_t* a0_ptr = a;
    size_t k = kc;

    // Strategy 9: Software pipeline scalar activation loads to decouple from vector execution.
    // Pre-load initial weight vector and first pair of scalar activations.
    vint8m2_t vb0;
    int16_t a0, a1;
    if (k > 0) {
      vb0 = __riscv_vle8_v_i8m2(w_weights, vl);
      w_weights += 16;
      a0 = (int16_t) a0_ptr[0];
      a1 = (int16_t) a0_ptr[1];
    }

    // Main inner loop: Unrolled to process 8 k-elements (4 vector weight bytes, 8 nibbles).
    while (k >= 8) {
      // Step 1: Hoist next weight load and next scalar pair (a2, a3) while computing (a0, a1).
      int16_t a2 = (int16_t) a0_ptr[2];
      int16_t a3 = (int16_t) a0_ptr[3];
      vint8m2_t vb1 = __riscv_vle8_v_i8m2(w_weights, vl); w_weights += 16;
      
      vint16m4_t w0_lo = __riscv_vsext_vf2_i16m4(__riscv_vsll_vx_i8m2(vb0, 4, vl), vl);
      vint16m4_t w0_hi = __riscv_vsext_vf2_i16m4(__riscv_vand_vx_i8m2(vb0, (int8_t) 0xF0, vl), vl);
      vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, a0, w0_lo, vl);
      vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, a1, w0_hi, vl);

      // Step 2: Hoist next weight load and next scalar pair (a4, a5) while computing (a2, a3).
      int16_t a4 = (int16_t) a0_ptr[4];
      int16_t a5 = (int16_t) a0_ptr[5];
      vint8m2_t vb2 = __riscv_vle8_v_i8m2(w_weights, vl); w_weights += 16;
      
      vint16m4_t w1_lo = __riscv_vsext_vf2_i16m4(__riscv_vsll_vx_i8m2(vb1, 4, vl), vl);
      vint16m4_t w1_hi = __riscv_vsext_vf2_i16m4(__riscv_vand_vx_i8m2(vb1, (int8_t) 0xF0, vl), vl);
      vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, a2, w1_lo, vl);
      vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, a3, w1_hi, vl);

      // Step 3: Hoist next weight load and final scalar pair (a6, a7) while computing (a4, a5).
      int16_t a6 = (int16_t) a0_ptr[6];
      int16_t a7 = (int16_t) a0_ptr[7];
      vint8m2_t vb3 = __riscv_vle8_v_i8m2(w_weights, vl); w_weights += 16;
      
      vint16m4_t w2_lo = __riscv_vsext_vf2_i16m4(__riscv_vsll_vx_i8m2(vb2, 4, vl), vl);
      vint16m4_t w2_hi = __riscv_vsext_vf2_i16m4(__riscv_vand_vx_i8m2(vb2, (int8_t) 0xF0, vl), vl);
      vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, a4, w2_lo, vl);
      vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, a5, w2_hi, vl);

      // Step 4: Advance pointers and pre-load for the next iteration (or tail loop).
      a0_ptr += 8;
      k -= 8;

      if (k > 0) {
        vb0 = __riscv_vle8_v_i8m2(w_weights, vl); w_weights += 16;
        a0 = (int16_t) a0_ptr[0];
        a1 = (int16_t) a0_ptr[1];
      }

      vint16m4_t w3_lo = __riscv_vsext_vf2_i16m4(__riscv_vsll_vx_i8m2(vb3, 4, vl), vl);
      vint16m4_t w3_hi = __riscv_vsext_vf2_i16m4(__riscv_vand_vx_i8m2(vb3, (int8_t) 0xF0, vl), vl);
      vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, a6, w3_lo, vl);
      vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, a7, w3_hi, vl);
    }

    // Tail loop: processes remaining pairs of k-elements using same hoisting pattern.
    while (k > 0) {
      vint16m4_t w_lo = __riscv_vsext_vf2_i16m4(__riscv_vsll_vx_i8m2(vb0, 4, vl), vl);
      vint16m4_t w_hi = __riscv_vsext_vf2_i16m4(__riscv_vand_vx_i8m2(vb0, (int8_t) 0xF0, vl), vl);
      
      // Keep current values for immediate use.
      int16_t a0_curr = a0;
      int16_t a1_curr = a1;

      a0_ptr += 2;
      k -= 2;

      // Hoist loads for the next tail iteration to overlap with current vwmacc.
      if (k > 0) {
        vb0 = __riscv_vle8_v_i8m2(w_weights, vl); w_weights += 16;
        a0 = (int16_t) a0_ptr[0];
        a1 = (int16_t) a0_ptr[1];
      }

      vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, a0_curr, w_lo, vl);
      vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, a1_curr, w_hi, vl);
    }

    // Merge accumulators: adjust scaling for vacc0 and sum with vacc1.
    // Both nibbles stay scaled by 16 (high nibble masked with 0xF0, as Neon does), so the sum is shifted
    // once: (A + 16H) >> 4.
    vint32m8_t vacc = __riscv_vsra_vx_i32m8(__riscv_vadd_vv_i32m8(vacc0, vacc1, vl), 4, vl);

    // Final conversion and output processing.
    vfloat32m8_t vout = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
    const float* w_scales = (const float*)w_weights;
    const float* w_biases = w_scales + 16;
    
    vfloat32m8_t vfilter_output_scale = __riscv_vle32_v_f32m8(w_scales, vl);
    vout = __riscv_vfmul_vf_f32m8(vout, inv_scale, vl);
    vfloat32m8_t vbias = __riscv_vle32_v_f32m8(w_biases, vl);

    // Output = (vout * scale) + bias
    vout = __riscv_vfmacc_vv_f32m8(vbias, vout, vfilter_output_scale, vl);

    // Min/Max Clamping.
    vout = __riscv_vfmax_vf_f32m8(vout, min_f, vl);
    vout = __riscv_vfmin_vf_f32m8(vout, max_f, vl);

    __riscv_vse32_v_f32m8(c, vout, vl);

    // Advance pointers to next 16-column block.
    c = (float*) ((uintptr_t) c + cn_stride);
    w = (const void*) (w_biases + 16);
    nc -= vl;
  }
}

