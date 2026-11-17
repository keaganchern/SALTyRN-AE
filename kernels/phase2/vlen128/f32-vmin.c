#include <riscv_vector.h>
#include <stddef.h>
#include <assert.h>

struct xnn_f32_default_params;

void test_rvv(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  if (n == 0) return;

  // Use vsetvlmax once to define the steady-state configuration.
  // Using LMUL=8 maximizes the chime length, which helps hide the 4-stage 
  // FMA/Arithmetic pipeline latency on Saturn's short-vector architecture.
  const size_t vl = __riscv_vsetvlmax_e32m8();

  // If the workload is larger than a single vector, enter the software-pipelined steady-state.
  if (n >= vl) {
    // Prime the pipeline: initial loads. 
    // These occupy the Load Sequencer (VLS).
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    input_a += vl;
    input_b += vl;
    size_t processed = vl;

    // Steady-state loop: processed + vl ensures we always have a full next chunk to load.
    while (processed + vl <= n) {
      // Interleave vector ops across different sequencers to leverage Saturn's DAE chaining.
      // Load Sequencer (VLS) starts fetching the next chunk.
      vfloat32m8_t va_next = __riscv_vle32_v_f32m8(input_a, vl);
      
      // Execute Sequencer (VXS) processes the current chunk.
      vfloat32m8_t vacc = __riscv_vfmin_vv_f32m8(va, vb, vl);
      input_a += vl; // Interleave scalar bookkeeping

      // VLS continues fetching.
      vfloat32m8_t vb_next = __riscv_vle32_v_f32m8(input_b, vl);
      input_b += vl;

      // Store Sequencer (VSS) writes back results from the previous Execute operation.
      __riscv_vse32_v_f32m8(output, vacc, vl);
      output += vl;

      // Advance registers using standard C assignment (zero-cost in the compiler).
      va = va_next;
      vb = vb_next;
      processed += vl;
    }

    // Drain the final full chunk processed by the pipeline
    vfloat32m8_t vacc = __riscv_vfmin_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
    n -= processed;
  }

  // Strategy 18: Add a fast-path early exit.
  // If the workload was a perfect multiple of vl, we exit immediately to avoid
  // the overhead of the tail-handling logic and associated branch evaluation.
  if (n == 0) return;

  // Handle remaining elements (the tail) unconditionally.
  // Since we have already exited if n == 0, we can skip the conditional check.
  size_t vlt = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t va_tail = __riscv_vle32_v_f32m8(input_a, vlt);
  vfloat32m8_t vb_tail = __riscv_vle32_v_f32m8(input_b, vlt);
  vfloat32m8_t vacc_tail = __riscv_vfmin_vv_f32m8(va_tail, vb_tail, vlt);
  __riscv_vse32_v_f32m8(output, vacc_tail, vlt);
}
