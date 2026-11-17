#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#ifndef XNN_U8_MINMAX_PARAMS_DEFINED
#define XNN_U8_MINMAX_PARAMS_DEFINED
struct xnn_u8_minmax_params {
  struct {
    uint8_t min;
    uint8_t max;
  } scalar;
};
#endif

void test_rvv(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_u8_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint8_t min = (uint8_t) params->scalar.min;
  const uint8_t max = (uint8_t) params->scalar.max;

  // Unrolled hot loop at fixed vlmax
  size_t vlmax = __riscv_vsetvlmax_e8m8();
  while (batch >= 2 * vlmax) {
    // Interleave instructions across VLS (load), VXS (execute), and VSS (store) sequencers.
    // This allows the backend to overlap memory access and computation via Decoupled Access-Execute.
    
    // Start Load 1 (VLS)
    vuint8m8_t vacc1 = __riscv_vle8_v_u8m8(input, vlmax);
    
    // Max 1 (VXS) can chain from Load 1 as soon as the first element-group is ready.
    vacc1 = __riscv_vmaxu_vx_u8m8(vacc1, min, vlmax);
    
    // Start Load 2 (VLS) while VXS is busy with Block 1.
    vuint8m8_t vacc2 = __riscv_vle8_v_u8m8(input + vlmax, vlmax);
    
    // Continue VXS pipeline for Block 1.
    vacc1 = __riscv_vminu_vx_u8m8(vacc1, max, vlmax);
    
    // Max 2 (VXS) can chain from Load 2.
    vacc2 = __riscv_vmaxu_vx_u8m8(vacc2, min, vlmax);
    
    // Store 1 (VSS) can chain from Min 1. 
    // VSS, VXS, and VLS are now potentially all active.
    __riscv_vse8_v_u8m8(output, vacc1, vlmax);
    
    // Complete VXS pipeline for Block 2.
    vacc2 = __riscv_vminu_vx_u8m8(vacc2, max, vlmax);
    
    // Store 2 (VSS) chains from Min 2.
    __riscv_vse8_v_u8m8(output + vlmax, vacc2, vlmax);

    input += 2 * vlmax;
    output += 2 * vlmax;
    batch -= 2 * vlmax;
  }

  // VLA tail loop for the remainder
  size_t vl;
  for (; batch > 0; batch -= vl) {
    vl = __riscv_vsetvl_e8m8(batch);
    
    vuint8m8_t vacc = __riscv_vle8_v_u8m8(input, vl);

    vacc = __riscv_vmaxu_vx_u8m8(vacc, min, vl);
    vacc = __riscv_vminu_vx_u8m8(vacc, max, vl);

    __riscv_vse8_v_u8m8(output, vacc, vl);

    input += vl;
    output += vl;
  }
}
