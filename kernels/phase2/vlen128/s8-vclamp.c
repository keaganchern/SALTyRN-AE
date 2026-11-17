#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

struct xnn_s8_minmax_params {
  struct {
    int8_t min;
    int8_t max;
  } scalar;
};

/**
 * Saturn-optimized s8 minmax using Strategy 13: Distance-based Chaining Optimization.
 * 
 * Saturn is a Decoupled Access-Execute (DAE) machine. To maximize throughput, 
 * we must ensure that the Load (VLS), Execute (VXS), and Store (VSS) sequencers 
 * overlap their execution via chaining. By interleaving scalar bookkeeping instructions 
 * between the load and the first dependent arithmetic instruction, we provide a 
 * "head start" to the Load sequencer. This allows the memory system to begin 
 * returning data and priming the chaining buffers before the Execute unit requires 
 * the first element group, reducing pipeline bubbles.
 */
void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int8_t output_min = (int8_t) params->scalar.min;
  const int8_t output_max = (int8_t) params->scalar.max;

  for (; batch > 0; ) {
    // Determine vector length for this iteration.
    // Saturn is a short-vector machine; using LMUL=8 (m8) maximizes chime length,
    // which helps hide pipeline latencies of the FUs and reduces scalar pressure.
    const size_t vl = __riscv_vsetvl_e8m8(batch);
    
    // 1. Dispatch Vector Load (VLS sequencer).
    // Vector instructions in Saturn are non-speculative and execute post-commit.
    vint8m8_t vacc = __riscv_vle8_v_i8m8(input, vl);

    // 2. Strategy 13: Distance-based Chaining Optimization.
    // We insert independent scalar bookkeeping between the load and compute.
    // The scalar core can execute these increments while the decoupled 
    // vector load unit is busy fetching data from memory.
    int8_t* current_output = output;
    input += vl;
    output += vl;
    batch -= vl;

    // 3. Dispatch Arithmetic (VXS sequencer).
    // These instructions will chain with the data arriving from the VLS unit.
    // vmax/vmin use the Integer Pipe (VXS), which is efficient for 1-stage ops.
    vacc = __riscv_vmax_vx_i8m8(vacc, output_min, vl);
    vacc = __riscv_vmin_vx_i8m8(vacc, output_max, vl);

    // 4. Dispatch Vector Store (VSS sequencer).
    // This chains with the result of the VXS sequencer.
    __riscv_vse8_v_i8m8(current_output, vacc, vl);
  }
}
