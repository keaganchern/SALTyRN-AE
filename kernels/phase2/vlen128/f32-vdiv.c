#include <riscv_vector.h>
#include <stddef.h>
#include <assert.h>

void test_rvv(
    size_t batch,
    const float* restrict input_a,
    const float* restrict input_b,
    float* restrict output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  // Using LMUL=4 provides a balance between long chimes (hiding latency) and 
  // register pressure. With double buffering, LMUL=4 uses ~20 registers, 
  // avoiding spilling to memory on Saturn's 32-register RF.
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  size_t i = 0;

  if (n >= vlmax) {
    // Prologue: initial load for the first iteration.
    // Loads utilize the VLS (Load) sequencer.
    vfloat32m4_t va = __riscv_vle32_v_f32m4(input_a, vlmax);
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(input_b, vlmax);

    // Main double-buffered loop.
    // Applying 'restrict' to pointers ensures the compiler knows 'output' does 
    // not alias with 'input_a' or 'input_b'. On Saturn, this allows the VLS 
    // sequencer to issue the next iteration's loads while the current 
    // iteration's store is still pending in the VSS sequencer.
    for (; i + 2 * vlmax <= n; i += vlmax) {
      // Load next iteration's data (VLS).
      // These can execute concurrently with the division (VXS) of the current block.
      vfloat32m4_t va_next = __riscv_vle32_v_f32m4(input_a + i + vlmax, vlmax);
      vfloat32m4_t vb_next = __riscv_vle32_v_f32m4(input_b + i + vlmax, vlmax);
      
      // Execute division (VXS).
      // On Saturn, vfdiv is iterative (1 elem/cycle). Double buffering helps
      // hide the significant latency of this unit.
      vfloat32m4_t vres = __riscv_vfdiv_vv_f32m4(va, vb, vlmax);
      
      // Store current result (VSS).
      // Chaining occurs between the VXS sequencer and the VSS sequencer.
      __riscv_vse32_v_f32m4(output + i, vres, vlmax);
      
      // Rotate buffers using standard C assignment.
      va = va_next;
      vb = vb_next;
    }

    // Epilogue: process and store the last full block.
    vfloat32m4_t vres = __riscv_vfdiv_vv_f32m4(va, vb, vlmax);
    __riscv_vse32_v_f32m4(output + i, vres, vlmax);
    i += vlmax;
  }

  // Handle final fractional vector block (tail).
  if (i < n) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t va = __riscv_vle32_v_f32m4(input_a + i, vl);
    vfloat32m4_t vb = __riscv_vle32_v_f32m4(input_b + i, vl);
    vfloat32m4_t vres = __riscv_vfdiv_vv_f32m4(va, vb, vl);
    __riscv_vse32_v_f32m4(output + i, vres, vl);
  }
}
