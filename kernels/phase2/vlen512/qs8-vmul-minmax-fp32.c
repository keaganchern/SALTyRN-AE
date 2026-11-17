// As Neon: prod * scale, rounded to an integer, then out_zp added (int16 saturating add after the convert).
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params* restrict params)
{
  if (params->scalar.output_zero_point == 0) {

    assert(batch != 0);
    assert(batch % sizeof(int8_t) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const int8_t a_zero_point = params->scalar.a_zero_point;
    const int8_t b_zero_point = params->scalar.b_zero_point;
    const float scale = params->scalar.scale;
    const int16_t output_zero_point = params->scalar.output_zero_point;
    const int8_t output_min = params->scalar.output_min;
    const int8_t output_max = params->scalar.output_max;

    // Pre-convert the scalar integer zero point to a float.

    // Pre-splat the output zero point to a vector register to avoid splatting inside the loop.
    // We use vlmax because this vector is read-only and reused across iterations with potentially varying vl.
    // The max vector length of e32m8 is structurally identical to e8m2.
    const size_t vlmax = __riscv_vsetvlmax_e8m2();

    // Fast-path for symmetric zero points: Skips subtractions and uses 16-bit intermediate products
    if (a_zero_point == 0 && b_zero_point == 0) {
      // Loop unswitching based on whether we can rely purely on vnclip's natural saturation.
      if (output_min == -128 && output_max == 127) {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          // Prologue: Compute chunk N+1 (1st chunk)
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            // Load chunk N+2
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            // Compute chunk N+1
            const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_out_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);

            // Store chunk N
            __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          
            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          // Epilogue: Compute the last chunk and store the remaining two chunks
          const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_out_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          // Fallback if data is too small to form a pipeline
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }

      } else {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          vout_store = __riscv_vmax_vx_i8m2(vout_store, output_min, vl);
          vout_store = __riscv_vmin_vx_i8m2(vout_store, output_max, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_out_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);
            vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
            vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          
            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_out_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);
          vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
          vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
          vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }
      }
    } else {
      // Generic path for asymmetric or non-zero input zero-points
      if (output_min == -128 && output_max == 127) {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
            const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
            const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);

            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }

      } else {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          vout_store = __riscv_vmax_vx_i8m2(vout_store, output_min, vl);
          vout_store = __riscv_vmin_vx_i8m2(vout_store, output_max, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
            const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
            const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);
            vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
            vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);

            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_new = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);
          vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
          vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
          vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }
      }
    }
  } else {   // round the product first, add the zero point as an integer

    assert(batch != 0);
    assert(batch % sizeof(int8_t) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const int8_t a_zero_point = params->scalar.a_zero_point;
    const int8_t b_zero_point = params->scalar.b_zero_point;
    const float scale = params->scalar.scale;
    const int16_t output_zero_point = params->scalar.output_zero_point;
    const int8_t output_min = params->scalar.output_min;
    const int8_t output_max = params->scalar.output_max;



    // Fast-path for symmetric zero points: Skips subtractions and uses 16-bit intermediate products
    if (a_zero_point == 0 && b_zero_point == 0) {
      // Loop unswitching based on whether we can rely purely on vnclip's natural saturation.
      if (output_min == -128 && output_max == 127) {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          // Prologue: Compute chunk N+1 (1st chunk)
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            // Load chunk N+2
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            // Compute chunk N+1
            const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_out_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);

            // Store chunk N
            __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          
            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          // Epilogue: Compute the last chunk and store the remaining two chunks
          const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_out_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          // Fallback if data is too small to form a pipeline
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }

      } else {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          vout_store = __riscv_vmax_vx_i8m2(vout_store, output_min, vl);
          vout_store = __riscv_vmin_vx_i8m2(vout_store, output_max, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_out_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);
            vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
            vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          
            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vacc16_new = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfwcvt_f_x_v_f32m8(vacc16_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_out_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_out_new, 0, __RISCV_VXRM_RNU, vl);
          vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
          vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vacc16 = __riscv_vwmul_vv_i16m4(va, vb, vl);
          vfloat32m8_t vfpacc = __riscv_vfwcvt_f_x_v_f32m8(vacc16, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16_out = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16_out, 0, __RISCV_VXRM_RNU, vl);
          vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
          vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }
      }
    } else {
      // Generic path for asymmetric or non-zero input zero-points
      if (output_min == -128 && output_max == 127) {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
            const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
            const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);

            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }

      } else {
        size_t vl = __riscv_vsetvl_e8m2(batch);
        vint8m2_t va = __riscv_vle8_v_i8m2(input_a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(input_b, vl);
        input_a += vl;
        input_b += vl;
        batch -= vl;

        if (batch > 0) {
          size_t next_vl = __riscv_vsetvl_e8m2(batch);
          vint8m2_t va_next = __riscv_vle8_v_i8m2(input_a, next_vl);
          vint8m2_t vb_next = __riscv_vle8_v_i8m2(input_b, next_vl);
          input_a += next_vl;
          input_b += next_vl;
          batch -= next_vl;

          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_store = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          vout_store = __riscv_vmax_vx_i8m2(vout_store, output_min, vl);
          vout_store = __riscv_vmin_vx_i8m2(vout_store, output_max, vl);
          size_t vl_store = vl;

          va = va_next;
          vb = vb_next;
          vl = next_vl;

          while (batch > 0) {
            size_t next_next_vl = __riscv_vsetvl_e8m2(batch);
            vint8m2_t va_next_next = __riscv_vle8_v_i8m2(input_a, next_next_vl);
            vint8m2_t vb_next_next = __riscv_vle8_v_i8m2(input_b, next_next_vl);

            const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
            const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
            const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
            vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
            vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
            const vint16m4_t vacc16_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
            vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);
            vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
            vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

            __riscv_vse8_v_i8m2(output, vout_store, vl_store);

            output += vl_store;
            input_a += next_next_vl;
            input_b += next_next_vl;
            batch -= next_next_vl;

            vout_store = vout_new;
            vl_store = vl;
            va = va_next_next;
            vb = vb_next_next;
            vl = next_next_vl;
          }

          const vint16m4_t vxa_new = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb_new = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc_new = __riscv_vwmul_vv_i32m8(vxa_new, vxb_new, vl);
          vfloat32m8_t vfpacc_new = __riscv_vfcvt_f_x_v_f32m8(vacc_new, vl);
          vfpacc_new = __riscv_vfmul_vf_f32m8(vfpacc_new, scale, vl);
          const vint16m4_t vacc16_new = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc_new, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout_new = __riscv_vnclip_wx_i8m2(vacc16_new, 0, __RISCV_VXRM_RNU, vl);
          vout_new = __riscv_vmax_vx_i8m2(vout_new, output_min, vl);
          vout_new = __riscv_vmin_vx_i8m2(vout_new, output_max, vl);

          __riscv_vse8_v_i8m2(output, vout_store, vl_store);
          output += vl_store;
          __riscv_vse8_v_i8m2(output, vout_new, vl);

        } else {
          const vint16m4_t vxa = __riscv_vwsub_vx_i16m4(va, a_zero_point, vl);
          const vint16m4_t vxb = __riscv_vwsub_vx_i16m4(vb, b_zero_point, vl);
          const vint32m8_t vacc = __riscv_vwmul_vv_i32m8(vxa, vxb, vl);
          vfloat32m8_t vfpacc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
          vfpacc = __riscv_vfmul_vf_f32m8(vfpacc, scale, vl);
          const vint16m4_t vacc16 = __riscv_vsadd_vx_i16m4(__riscv_vfncvt_x_f_w_i16m4_rm(vfpacc, __RISCV_FRM_RNE, vl), output_zero_point, vl);
          vint8m2_t vout = __riscv_vnclip_wx_i8m2(vacc16, 0, __RISCV_VXRM_RNU, vl);
          vout = __riscv_vmax_vx_i8m2(vout, output_min, vl);
          vout = __riscv_vmin_vx_i8m2(vout, output_max, vl);
          __riscv_vse8_v_i8m2(output, vout, vl);
        }
      }
    }
  }
}
