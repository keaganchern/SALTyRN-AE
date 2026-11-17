#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const uint8_t* input,
    float* output,
    const struct xnn_qu8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float scale = params->scalar.scale;
  const uint8_t zero_point = (uint8_t)params->scalar.zero_point;

  size_t vlmax = __riscv_vsetvlmax_e32m8();
  const size_t step = 2 * vlmax;

  while (batch >= step) {
    // Stage 1: Parallel Loads (VLS Sequencer)
    vuint8m2_t vxA = __riscv_vle8_v_u8m2(input, vlmax);
    vuint8m2_t vxB = __riscv_vle8_v_u8m2(input + vlmax, vlmax);

    // Stage 2-4: VXS Sub-Pipeline Interleaving
    // Interleave instructions targeting different execution units to avoid single-queue stalls
    
    // IntPipe (Block A)
    vuint16m4_t vxuA = __riscv_vwsubu_vx_u16m4(vxA, zero_point, vlmax);
    vint16m4_t vxiA = __riscv_vreinterpret_v_u16m4_i16m4(vxuA);
    
    // FPConvPipe (Block A)
    vfloat32m8_t vxfA = __riscv_vfwcvt_f_x_v_f32m8(vxiA, vlmax);
    
    // IntPipe (Block B) - Issues while FPConvPipe processes Block A
    vuint16m4_t vxuB = __riscv_vwsubu_vx_u16m4(vxB, zero_point, vlmax);
    vint16m4_t vxiB = __riscv_vreinterpret_v_u16m4_i16m4(vxuB);
    
    // FMAPipe (Block A)
    vfloat32m8_t vyA = __riscv_vfmul_vf_f32m8(vxfA, scale, vlmax);
    
    // FPConvPipe (Block B) - Issues while FMAPipe processes Block A
    vfloat32m8_t vxfB = __riscv_vfwcvt_f_x_v_f32m8(vxiB, vlmax);
    
    // Store Block A (VSS Sequencer) - Dispatched while VXS continues with Block B
    __riscv_vse32_v_f32m8(output, vyA, vlmax);

    // FMAPipe (Block B)
    vfloat32m8_t vyB = __riscv_vfmul_vf_f32m8(vxfB, scale, vlmax);

    // Store Block B (VSS Sequencer)
    __riscv_vse32_v_f32m8(output + vlmax, vyB, vlmax);

    input += step;
    output += step;
    batch -= step;
  }

  // Standard strip-mining tail loop for remaining elements (< 2*vlmax)
  while (batch > 0) {
    size_t vl = __riscv_vsetvl_e8m2(batch);
    
    vuint8m2_t vx = __riscv_vle8_v_u8m2(input, vl);
    vuint16m4_t vxu = __riscv_vwsubu_vx_u16m4(vx, zero_point, vl);
    vint16m4_t vxi = __riscv_vreinterpret_v_u16m4_i16m4(vxu);
    vfloat32m8_t vxf = __riscv_vfwcvt_f_x_v_f32m8(vxi, vl);
    vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vxf, scale, vl);
    __riscv_vse32_v_f32m8(output, vy, vl);

    input += vl;
    output += vl;
    batch -= vl;
  }
}
