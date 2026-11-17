#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

// -x*128 (or (izp - x)*128) times -multiplier, rounded once, as the Neon kernel does.

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

  const int8_t input_zero_point = (int8_t)params->scalar.input_zero_point;
  // Negated: -multiplier fits int16 for every valid multiplier (1..32768); (int16_t)32768 wraps to -32768.
  const int16_t multiplier = (int16_t)(-params->scalar.multiplier);
  const int16_t output_zero_point = params->scalar.output_zero_point;

  size_t vlmax = __riscv_vsetvlmax_e8m4();
  size_t i = 0;

  if (input_zero_point == 0) {
    // -x*128 times -multiplier: x*128*multiplier, one rounding (vwmul by -128 costs what vwmulsu by 128 did)
    if (output_zero_point == 0) {
      for (; i + 2 * vlmax <= batch; i += 2 * vlmax) {
        vint8m4_t vx1 = __riscv_vle8_v_i8m4(input + i, vlmax);
        vint16m8_t vacc1 = __riscv_vwmul_vx_i16m8(vx1, -128, vlmax);

        vint8m4_t vx2 = __riscv_vle8_v_i8m4(input + i + vlmax, vlmax);
        vacc1 = __riscv_vsmul_vx_i16m8(vacc1, multiplier, __RISCV_VXRM_RNU, vlmax);

        vint16m8_t vacc2 = __riscv_vwmul_vx_i16m8(vx2, -128, vlmax);
        vint8m4_t vy1 = __riscv_vnclip_wx_i8m4(vacc1, 0, __RISCV_VXRM_RNU, vlmax);

        __riscv_vse8_v_i8m4(output + i, vy1, vlmax);
        vacc2 = __riscv_vsmul_vx_i16m8(vacc2, multiplier, __RISCV_VXRM_RNU, vlmax);

        vint8m4_t vy2 = __riscv_vnclip_wx_i8m4(vacc2, 0, __RISCV_VXRM_RNU, vlmax);
        __riscv_vse8_v_i8m4(output + i + vlmax, vy2, vlmax);
      }

      for (; i < batch; ) {
        size_t vl = __riscv_vsetvl_e8m4(batch - i);
        vint8m4_t vx = __riscv_vle8_v_i8m4(input + i, vl);
        vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx, -128, vl);
        vacc = __riscv_vsmul_vx_i16m8(vacc, multiplier, __RISCV_VXRM_RNU, vl);
        vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m4(output + i, vy, vl);
        i += vl;
      }
    } else {   // output zero point added after the rounding multiply
      for (; i + 2 * vlmax <= batch; i += 2 * vlmax) {
        vint8m4_t vx1 = __riscv_vle8_v_i8m4(input + i, vlmax);
        vint16m8_t vacc1 = __riscv_vwmul_vx_i16m8(vx1, -128, vlmax);

        vint8m4_t vx2 = __riscv_vle8_v_i8m4(input + i + vlmax, vlmax);
        vacc1 = __riscv_vsmul_vx_i16m8(vacc1, multiplier, __RISCV_VXRM_RNU, vlmax);

        vint16m8_t vacc2 = __riscv_vwmul_vx_i16m8(vx2, -128, vlmax);
        vacc1 = __riscv_vsadd_vx_i16m8(vacc1, output_zero_point, vlmax);

        vacc2 = __riscv_vsmul_vx_i16m8(vacc2, multiplier, __RISCV_VXRM_RNU, vlmax);
        vint8m4_t vy1 = __riscv_vnclip_wx_i8m4(vacc1, 0, __RISCV_VXRM_RNU, vlmax);

        __riscv_vse8_v_i8m4(output + i, vy1, vlmax);
        vacc2 = __riscv_vsadd_vx_i16m8(vacc2, output_zero_point, vlmax);

        vint8m4_t vy2 = __riscv_vnclip_wx_i8m4(vacc2, 0, __RISCV_VXRM_RNU, vlmax);
        __riscv_vse8_v_i8m4(output + i + vlmax, vy2, vlmax);
      }

      for (; i < batch; ) {
        size_t vl = __riscv_vsetvl_e8m4(batch - i);
        vint8m4_t vx = __riscv_vle8_v_i8m4(input + i, vl);
        vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx, -128, vl);
        vacc = __riscv_vsmul_vx_i16m8(vacc, multiplier, __RISCV_VXRM_RNU, vl);
        vacc = __riscv_vsadd_vx_i16m8(vacc, output_zero_point, vl);
        vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_i8m4(output + i, vy, vl);
        i += vl;
      }
    }
  } else {
    // (izp - x)*128 times -multiplier: (x - izp)*128*multiplier rounded once, as in Neon (one extra e16m8 op: vmul)
    for (; i + 2 * vlmax <= batch; i += 2 * vlmax) {
      vint8m4_t vx1 = __riscv_vle8_v_i8m4(input + i, vlmax);
      vint16m8_t vacc1 = __riscv_vmul_vx_i16m8(__riscv_vwsub_vx_i16m8(vx1, input_zero_point, vlmax), -128, vlmax);

      vint8m4_t vx2 = __riscv_vle8_v_i8m4(input + i + vlmax, vlmax);
      vacc1 = __riscv_vsmul_vx_i16m8(vacc1, multiplier, __RISCV_VXRM_RNU, vlmax);

      vint16m8_t vacc2 = __riscv_vmul_vx_i16m8(__riscv_vwsub_vx_i16m8(vx2, input_zero_point, vlmax), -128, vlmax);
      vacc1 = __riscv_vsadd_vx_i16m8(vacc1, output_zero_point, vlmax);

      vacc2 = __riscv_vsmul_vx_i16m8(vacc2, multiplier, __RISCV_VXRM_RNU, vlmax);
      vint8m4_t vy1 = __riscv_vnclip_wx_i8m4(vacc1, 0, __RISCV_VXRM_RNU, vlmax);

      __riscv_vse8_v_i8m4(output + i, vy1, vlmax);
      vacc2 = __riscv_vsadd_vx_i16m8(vacc2, output_zero_point, vlmax);

      vint8m4_t vy2 = __riscv_vnclip_wx_i8m4(vacc2, 0, __RISCV_VXRM_RNU, vlmax);
      __riscv_vse8_v_i8m4(output + i + vlmax, vy2, vlmax);
    }

    for (; i < batch; ) {
      size_t vl = __riscv_vsetvl_e8m4(batch - i);
      vint8m4_t vx = __riscv_vle8_v_i8m4(input + i, vl);
      vint16m8_t vacc = __riscv_vmul_vx_i16m8(__riscv_vwsub_vx_i16m8(vx, input_zero_point, vl), -128, vl);
      vacc = __riscv_vsmul_vx_i16m8(vacc, multiplier, __RISCV_VXRM_RNU, vl);
      vacc = __riscv_vsadd_vx_i16m8(vacc, output_zero_point, vl);
      vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m4(output + i, vy, vl);
      i += vl;
    }
  }
}
