#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

struct xnn_qs8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  // Scalar parameter extraction
  const int8_t izp = (int8_t) params->scalar.input_zero_point;
  // Multipliers negated (-m fits int16 for every valid m, up to 32768); the operand is -(x - izp)*128.
  const int16_t positive_multiplier = (int16_t) (-params->scalar.positive_multiplier);
  const int16_t negative_multiplier = (int16_t) (-params->scalar.negative_multiplier);
  const int16_t ozp = (int16_t) params->scalar.output_zero_point;

  // Use LMUL=4 for 8-bit elements and LMUL=8 for 16-bit intermediate arithmetic to maximize 
  // Saturn's chime length, hiding the 4-stage FMA latency and pipeline depth.
  const size_t vlmax = __riscv_vsetvlmax_e8m4();
  vint16m8_t vpos_mult = __riscv_vmv_v_x_i16m8(positive_multiplier, vlmax);

  // We employ Software Pipelining (Double Buffering) to overlap the Vector Load-Store Unit (VLS) 
  // with the Vector Datapath (VXS).
  if (izp == 0 && ozp == 0) {
    size_t batch_rem = batch;
    size_t vl_cur = __riscv_vsetvl_e8m4(batch_rem);
    vint8m4_t vx_cur = __riscv_vle8_v_i8m4(input, vl_cur);
    input += vl_cur;
    batch_rem -= vl_cur;

    while (batch_rem > 0) {
      size_t vl_next = __riscv_vsetvl_e8m4(batch_rem);
      vint8m4_t vx_next = __riscv_vle8_v_i8m4(input, vl_next); // Prefetch next batch
      input += vl_next;
      batch_rem -= vl_next;

      vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, 0, vl_cur);
      vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx_cur, -128, vl_cur);   // -x*128
      vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
      vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
      vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);

      __riscv_vse8_v_i8m4(output, vy, vl_cur); // Store current batch
      output += vl_cur;
      
      vx_cur = vx_next;
      vl_cur = vl_next;
    }
    // Final tail compute
    vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, 0, vl_cur);
    vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx_cur, -128, vl_cur);   // -x*128
    vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
    vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
    vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);
    __riscv_vse8_v_i8m4(output, vy, vl_cur);

  } else if (izp == 0) {
    size_t batch_rem = batch;
    size_t vl_cur = __riscv_vsetvl_e8m4(batch_rem);
    vint8m4_t vx_cur = __riscv_vle8_v_i8m4(input, vl_cur);
    input += vl_cur;
    batch_rem -= vl_cur;

    while (batch_rem > 0) {
      size_t vl_next = __riscv_vsetvl_e8m4(batch_rem);
      vint8m4_t vx_next = __riscv_vle8_v_i8m4(input, vl_next);
      input += vl_next;
      batch_rem -= vl_next;

      vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, 0, vl_cur);
      vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx_cur, -128, vl_cur);   // -x*128
      vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
      vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
      vacc = __riscv_vsadd_vx_i16m8(vacc, ozp, vl_cur);
      vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);

      __riscv_vse8_v_i8m4(output, vy, vl_cur);
      output += vl_cur;
      
      vx_cur = vx_next;
      vl_cur = vl_next;
    }
    vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, 0, vl_cur);
    vint16m8_t vacc = __riscv_vwmul_vx_i16m8(vx_cur, -128, vl_cur);   // -x*128
    vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
    vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
    vacc = __riscv_vsadd_vx_i16m8(vacc, ozp, vl_cur);
    vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);
    __riscv_vse8_v_i8m4(output, vy, vl_cur);

  } else if (ozp == 0) {
    size_t batch_rem = batch;
    size_t vl_cur = __riscv_vsetvl_e8m4(batch_rem);
    vint8m4_t vx_cur = __riscv_vle8_v_i8m4(input, vl_cur);
    input += vl_cur;
    batch_rem -= vl_cur;

    while (batch_rem > 0) {
      size_t vl_next = __riscv_vsetvl_e8m4(batch_rem);
      vint8m4_t vx_next = __riscv_vle8_v_i8m4(input, vl_next);
      input += vl_next;
      batch_rem -= vl_next;

      vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, izp, vl_cur);
      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx_cur, izp, vl_cur);
      vacc = __riscv_vmul_vx_i16m8(vacc, -128, vl_cur);   // -(x - izp)*128
      vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
      vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
      vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);

      __riscv_vse8_v_i8m4(output, vy, vl_cur);
      output += vl_cur;
      
      vx_cur = vx_next;
      vl_cur = vl_next;
    }
    vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, izp, vl_cur);
    vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx_cur, izp, vl_cur);
    vacc = __riscv_vmul_vx_i16m8(vacc, -128, vl_cur);   // -(x - izp)*128
    vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
    vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
    vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);
    __riscv_vse8_v_i8m4(output, vy, vl_cur);

  } else {
    size_t batch_rem = batch;
    size_t vl_cur = __riscv_vsetvl_e8m4(batch_rem);
    vint8m4_t vx_cur = __riscv_vle8_v_i8m4(input, vl_cur);
    input += vl_cur;
    batch_rem -= vl_cur;

    while (batch_rem > 0) {
      size_t vl_next = __riscv_vsetvl_e8m4(batch_rem);
      vint8m4_t vx_next = __riscv_vle8_v_i8m4(input, vl_next);
      input += vl_next;
      batch_rem -= vl_next;

      vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, izp, vl_cur);
      vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx_cur, izp, vl_cur);
      vacc = __riscv_vmul_vx_i16m8(vacc, -128, vl_cur);   // -(x - izp)*128
      vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
      vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
      vacc = __riscv_vsadd_vx_i16m8(vacc, ozp, vl_cur);
      vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);

      __riscv_vse8_v_i8m4(output, vy, vl_cur);
      output += vl_cur;
      
      vx_cur = vx_next;
      vl_cur = vl_next;
    }
    vbool2_t vmask = __riscv_vmslt_vx_i8m4_b2(vx_cur, izp, vl_cur);
    vint16m8_t vacc = __riscv_vwsub_vx_i16m8(vx_cur, izp, vl_cur);
    vacc = __riscv_vmul_vx_i16m8(vacc, -128, vl_cur);   // -(x - izp)*128
    vint16m8_t vmult = __riscv_vmerge_vxm_i16m8(vpos_mult, negative_multiplier, vmask, vl_cur);
    vacc = __riscv_vsmul_vv_i16m8(vacc, vmult, __RISCV_VXRM_RNU, vl_cur);
    vacc = __riscv_vsadd_vx_i16m8(vacc, ozp, vl_cur);
    vint8m4_t vy = __riscv_vnclip_wx_i8m4(vacc, 0, __RISCV_VXRM_RDN, vl_cur);
    __riscv_vse8_v_i8m4(output, vy, vl_cur);
  }
}
