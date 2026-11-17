#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Calculate total number of elements.
  // The architecture summary notes that Saturn processes 1 element group (DLEN bits) per cycle.
  size_t n = batch / sizeof(float);
  if (n == 0) return;

  const float* ptr_in = input;
  float* ptr_out = output;

  // Strategy 13: Static VL Stripmining
  // We set the vector length to the maximum (vlmax) for SEW=32 and LMUL=8.
  // By using a constant vl for the main loop, we eliminate the vsetvl configuration 
  // bubble and reduce the scalar bookkeeping required in every iteration.
  const size_t vl = __riscv_vsetvlmax_e32m8();

  // Main loop: Process elements in vl-sized blocks.
  // Saturn's Decoupled Access-Execute (DAE) unit allows the VLS (Load), 
  // VXS (Arithmetic/Sqrt), and VSS (Store) sequencers to operate independently.
  // As the scalar core dispatches these instructions, the hardware automatically
  // overlaps the Load of block i+1 with the Sqrt of block i and the Store of block i-1.
  while (n >= vl) {
    // Unit-stride load (VLS sequencer)
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(ptr_in, vl);
    
    // Iterative square root (VXS sequencer). 
    // Although iterative and slow (1 elem/cycle), Saturn can overlap this with 
    // memory operations on other sequencers.
    vfloat32m8_t vy = __riscv_vfsqrt_v_f32m8(vx, vl);
    
    // Unit-stride store (VSS sequencer)
    __riscv_vse32_v_f32m8(ptr_out, vy, vl);

    // Static increments allow the scalar core to run ahead effectively.
    ptr_in += vl;
    ptr_out += vl;
    n -= vl;
  }

  // Tail handling: Process remaining elements where n < vlmax.
  if (n > 0) {
    size_t vlt = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(ptr_in, vlt);
    vfloat32m8_t vy = __riscv_vfsqrt_v_f32m8(vx, vlt);
    __riscv_vse32_v_f32m8(ptr_out, vy, vlt);
  }
}
