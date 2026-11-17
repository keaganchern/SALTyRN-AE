#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

void test_rvv(
    size_t batch,
    const float* input,
    uint16_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vexp_bias = UINT32_C(0x07800000);
  const float vscale_to_inf = 0x1.0p+112f;
  const uint32_t vexpw_max = UINT32_C(0x7F800000);
  const float vscale_to_zero = 0x1.0p-110f;
  const uint16_t vexph_mask = UINT16_C(0x7C00);
  const uint16_t vmanth_mask = UINT16_C(0x0FFF);
  const uint16_t vsignh_mask = UINT16_C(0x8000);
  const uint16_t vnanh = UINT16_C(0x7E00);

  size_t n = batch / sizeof(float);
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);

    // Initial load starts the VLS sequencer.
    // LMUL=8 provides long chimes to hide pipeline latency and overlap scalar work.
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    // Interleave scalar bookkeeping with vector execution.
    // Saturn's DAE design allows the scalar core to process these increments 
    // while the VU handles the computation pipeline.
    uint16_t* current_output = output;
    input += vl;
    output += vl;

    vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx, vl);
    vuint32m8_t vabsx_u32 = __riscv_vreinterpret_v_f32m8_u32m8(vabsx);

    n -= vl;

    vuint32m8_t vbias = __riscv_vadd_vx_u32m8(vabsx_u32, vexp_bias, vl);
    
    // First stage of scaling: FMAPipe (4 stages). 
    vfloat32m8_t vf = __riscv_vfmul_vf_f32m8(vabsx, vscale_to_inf, vl);
    
    // Strategy 17: VXS Pipeline Interleaving
    // Interleave the ShiftPipe computation for vsignh with FMAPipe instructions
    // to prevent the VXS issue queue from overflowing with back-to-back logic ops.
    vuint32m8_t vx_u32 = __riscv_vreinterpret_v_f32m8_u32m8(vx);
    vuint16m4_t vsignh = __riscv_vnsrl_wx_u16m4(vx_u32, 16, vl);
    
    // Evaluate x != x (IEEE-754) to detect NaNs via FPConvPipe (2 stages).
    vbool4_t vnanmaskw = __riscv_vmfne_vv_f32m8_b4(vx, vx, vl);

    vbias = __riscv_vand_vx_u32m8(vbias, vexpw_max, vl);

    // Interleave the BitwisePipe computation for vsignh
    vsignh = __riscv_vand_vx_u16m4(vsignh, vsignh_mask, vl);

    vfloat32m8_t vbias_f32 = __riscv_vreinterpret_v_u32m8_f32m8(vbias);
    vbias_f32 = __riscv_vfmax_vf_f32m8(vbias_f32, 2.0f, vl);
    
    // Fused multiply-accumulate to scale and add bias in one 4-stage pass.
    vf = __riscv_vfmacc_vf_f32m8(vbias_f32, vscale_to_zero, vf, vl);

    vuint32m8_t vf_u32 = __riscv_vreinterpret_v_f32m8_u32m8(vf);

    // Continue applying Strategy 17 by alternating ShiftPipe and BitwisePipe 
    // operations for vexph and vmanth rather than grouping all shifts together.
    vuint16m4_t vexph = __riscv_vnsrl_wx_u16m4(vf_u32, 13, vl);
    vexph = __riscv_vand_vx_u16m4(vexph, vexph_mask, vl);

    vuint16m4_t vmanth = __riscv_vnsrl_wx_u16m4(vf_u32, 0, vl);
    vmanth = __riscv_vand_vx_u16m4(vmanth, vmanth_mask, vl);

    vuint16m4_t vh = __riscv_vadd_vv_u16m4(vmanth, vexph, vl);
    vh = __riscv_vmerge_vxm_u16m4(vh, vnanh, vnanmaskw, vl);
    vh = __riscv_vor_vv_u16m4(vh, vsignh, vl);

    // Unit-stride store to close the DAE loop.
    __riscv_vse16_v_u16m4(current_output, vh, vl);
  }
}
