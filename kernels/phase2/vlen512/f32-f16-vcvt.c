#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

void test_rvv(
    size_t batch,
    const float* input,
    uint16_t* output,
    const void* params)
{
  (void)params;
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  if (n == 0) return;

  size_t i = 0;
  size_t vl0 = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t vx0 = __riscv_vle32_v_f32m8(input + i, vl0);
  i += vl0;

  // Double buffered loop: dynamically handles sizes and eliminates the tail loop
  while (i < n) {
    size_t vl1 = __riscv_vsetvl_e32m8(n - i);

    vuint32m8_t vu_x = __riscv_vreinterpret_v_f32m8_u32m8(vx0);
    vuint16m4_t vsignh = __riscv_vnsrl_wx_u16m4(vu_x, 16, vl0);
    vsignh = __riscv_vand_vx_u16m4(vsignh, UINT16_C(0x8000), vl0);

    vbool4_t vnanmask = __riscv_vmfne_vv_f32m8_b4(vx0, vx0, vl0);
    vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx0, vl0);

    vuint32m8_t vu_absx = __riscv_vreinterpret_v_f32m8_u32m8(vabsx);
    vuint32m8_t vbias = __riscv_vadd_vx_u32m8(vu_absx, UINT32_C(0x07800000), vl0);

    vfloat32m8_t vf = __riscv_vfmul_vf_f32m8(vabsx, 0x1.0p+112f, vl0);

    vbias = __riscv_vand_vx_u32m8(vbias, UINT32_C(0x7F800000), vl0);
    vbias = __riscv_vmaxu_vx_u32m8(vbias, UINT32_C(0x40000000), vl0);

    vf = __riscv_vfmadd_vf_f32m8(
        vf,
        0x1.0p-110f,
        __riscv_vreinterpret_v_u32m8_f32m8(vbias),
        vl0);

    vuint32m8_t vu_f = __riscv_vreinterpret_v_f32m8_u32m8(vf);

    vuint16m4_t vexph = __riscv_vnsrl_wx_u16m4(vu_f, 13, vl0);
    vuint16m4_t vmanth = __riscv_vncvt_x_x_w_u16m4(vu_f, vl0);

    // Overlap data movement and compute: vu_f is dead here, freeing 8 vector registers.
    // We now have enough space to safely load the next buffer without spilling VRF to memory.
    vfloat32m8_t vx1 = __riscv_vle32_v_f32m8(input + i, vl1);

    vexph = __riscv_vand_vx_u16m4(vexph, UINT16_C(0x7C00), vl0);
    vmanth = __riscv_vand_vx_u16m4(vmanth, UINT16_C(0x0FFF), vl0);

    vuint16m4_t vh = __riscv_vadd_vv_u16m4(vmanth, vexph, vl0);
    vh = __riscv_vmerge_vxm_u16m4(vh, UINT16_C(0x7E00), vnanmask, vl0);
    vh = __riscv_vor_vv_u16m4(vh, vsignh, vl0);

    __riscv_vse16_v_u16m4(output + i - vl0, vh, vl0);

    // Rotate buffers for the next iteration
    vx0 = vx1;
    vl0 = vl1;
    i += vl1;
  }

  // Drain the final buffer outside the loop
  vuint32m8_t vu_x = __riscv_vreinterpret_v_f32m8_u32m8(vx0);
  vuint16m4_t vsignh = __riscv_vnsrl_wx_u16m4(vu_x, 16, vl0);
  vsignh = __riscv_vand_vx_u16m4(vsignh, UINT16_C(0x8000), vl0);

  vbool4_t vnanmask = __riscv_vmfne_vv_f32m8_b4(vx0, vx0, vl0);
  vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx0, vl0);

  vuint32m8_t vu_absx = __riscv_vreinterpret_v_f32m8_u32m8(vabsx);
  vuint32m8_t vbias = __riscv_vadd_vx_u32m8(vu_absx, UINT32_C(0x07800000), vl0);

  vfloat32m8_t vf = __riscv_vfmul_vf_f32m8(vabsx, 0x1.0p+112f, vl0);

  vbias = __riscv_vand_vx_u32m8(vbias, UINT32_C(0x7F800000), vl0);
  vbias = __riscv_vmaxu_vx_u32m8(vbias, UINT32_C(0x40000000), vl0);

  vf = __riscv_vfmadd_vf_f32m8(
      vf,
      0x1.0p-110f,
      __riscv_vreinterpret_v_u32m8_f32m8(vbias),
      vl0);

  vuint32m8_t vu_f = __riscv_vreinterpret_v_f32m8_u32m8(vf);

  vuint16m4_t vexph = __riscv_vnsrl_wx_u16m4(vu_f, 13, vl0);
  vuint16m4_t vmanth = __riscv_vncvt_x_x_w_u16m4(vu_f, vl0);

  vexph = __riscv_vand_vx_u16m4(vexph, UINT16_C(0x7C00), vl0);
  vmanth = __riscv_vand_vx_u16m4(vmanth, UINT16_C(0x0FFF), vl0);

  vuint16m4_t vh = __riscv_vadd_vv_u16m4(vmanth, vexph, vl0);
  vh = __riscv_vmerge_vxm_u16m4(vh, UINT16_C(0x7E00), vnanmask, vl0);
  vh = __riscv_vor_vv_u16m4(vh, vsignh, vl0);

  __riscv_vse16_v_u16m4(output + i - vl0, vh, vl0);
}
