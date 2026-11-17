#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

void test_rvv(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  // Convert batch bytes to element count.
  size_t n = batch / sizeof(float);
  const float b = *input_b;

  // Use an end-pointer for loop termination to reduce index-based address calculations.
  const float* end_a = input_a + n;

  // Strategy: Use LMUL=8 to maximize chime length. On Saturn (a short-vector machine),
  // this hides the 4-stage FMA pipeline latency and reduces scalar fetch pressure.
  size_t vl = __riscv_vsetvl_e32m8(n);
  
  // Prologue: Load the first chunk (chunk i) to prime the software pipeline.
  // This utilizes the VLS (Load) sequencer.
  vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
  
  // Update pointers and track remaining elements to be prefetched.
  // Incrementing pointers directly is more efficient than base + index on Saturn.
  input_a += vl;
  size_t remaining = n - vl;

  // Main loop: Prefetch chunk i+1 while computing and storing chunk i.
  // This maximizes overlap between the independent Saturn sequencers (VLS, VXS, VSS).
  while (input_a < end_a) {
    // 1. Scalar Bookkeeping: Calculate next vector length.
    // Using a dedicated 'remaining' variable avoids costly pointer subtraction (end_a - input_a).
    size_t next_vl = __riscv_vsetvl_e32m8(remaining);

    // 2. Prefetch: Initiate next load (VLS sequencer).
    vfloat32m8_t va_next = __riscv_vle32_v_f32m8(input_a, next_vl);

    // 3. Compute: Process current chunk (VXS sequencer).
    vfloat32m8_t vacc = __riscv_vfmul_vf_f32m8(va, b, vl);

    // 4. Store: Commit current chunk to memory (VSS sequencer).
    // This store will chain with the vfmul arithmetic in the VXS sequencer.
    __riscv_vse32_v_f32m8(output, vacc, vl);

    // 5. Pointer Rotation: Update base pointers for the next iteration.
    // Saturn's decoupled scalar core performs these while vector units are busy.
    output += vl;
    input_a += next_vl;
    remaining -= next_vl;

    // 6. Register Rotation: Move the prefetched chunk into the current processing register.
    va = va_next;
    vl = next_vl;
  }

  // Epilogue: Process and store the final chunk.
  // This handles the last prefetched block (or the only block if n <= vlmax).
  vfloat32m8_t vacc_tail = __riscv_vfmul_vf_f32m8(va, b, vl);
  __riscv_vse32_v_f32m8(output, vacc_tail, vl);
}
