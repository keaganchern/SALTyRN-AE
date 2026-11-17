#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void test_rvv(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  if (batch == 0) return;

  // Hoist scalar parameters to registers.
  const int8_t zero_point = (int8_t)params->scalar.zero_point;
  const float scale = params->scalar.scale;

  // Prologue Stage 0: Load first chunk and update pointers/counters
  // Pointer bumping minimizes scalar calculation overhead in the hot path.
  size_t vl_curr = __riscv_vsetvl_e8m2(batch);
  vint8m2_t vx_curr = __riscv_vle8_v_i8m2(input, vl_curr);
  input += vl_curr;
  batch -= vl_curr;

  if (batch > 0) {
    // Prologue Stage 1: Load second chunk and compute the first chunk.
    // This allows the software pipeline to fully prime for VLS/VXS/VSS overlap.
    size_t vl_next = __riscv_vsetvl_e8m2(batch);
    vint8m2_t vx_next = __riscv_vle8_v_i8m2(input, vl_next);
    input += vl_next;
    batch -= vl_next;
    
    // Compute Chunk 0: Results in vy_prev
    // vhx is vint16m4_t, vy_prev is vfloat32m8_t. Using max LMUL hides latencies.
    vint16m4_t vhx = __riscv_vwsub_vx_i16m4(vx_curr, zero_point, vl_curr);
    vfloat32m8_t vy_prev = __riscv_vfwcvt_f_x_v_f32m8(vhx, vl_curr);
    vy_prev = __riscv_vfmul_vf_f32m8(vy_prev, scale, vl_curr);
    
    size_t vl_prev = vl_curr;
    vl_curr = vl_next;
    vx_curr = vx_next;
    
    // Steady-State: 3-Stage Software Pipeline
    // Stage 1 (VLS): Load N+1 | Stage 2 (VXS): Compute N | Stage 3 (VSS): Store N-1
    // Saturn's decoupled sequencers allow these three stages to execute concurrently.
    while (batch > 0) {
      vl_next = __riscv_vsetvl_e8m2(batch);
      
      // Stage 1 (VLS): Load next chunk
      vx_next = __riscv_vle8_v_i8m2(input, vl_next);
      input += vl_next;
      batch -= vl_next;
      
      // Stage 2 (VXS): Compute current chunk (vhx_i is local, hardware handles renaming)
      vint16m4_t vhx_i = __riscv_vwsub_vx_i16m4(vx_curr, zero_point, vl_curr);
      vfloat32m8_t vy_curr = __riscv_vfwcvt_f_x_v_f32m8(vhx_i, vl_curr);
      vy_curr = __riscv_vfmul_vf_f32m8(vy_curr, scale, vl_curr);
      
      // Stage 3 (VSS): Store previously computed chunk
      __riscv_vse32_v_f32m8(output, vy_prev, vl_prev);
      output += vl_prev;
      
      // Rotate pipeline state for next iteration
      vy_prev = vy_curr;
      vx_curr = vx_next;
      vl_prev = vl_curr;
      vl_curr = vl_next;
    }
    
    // Epilogue: Drain the remaining stages of the software pipeline
    // Compute last loaded chunk (Stage 2)
    vint16m4_t vhx_e = __riscv_vwsub_vx_i16m4(vx_curr, zero_point, vl_curr);
    vfloat32m8_t vy_curr_epi = __riscv_vfwcvt_f_x_v_f32m8(vhx_e, vl_curr);
    vy_curr_epi = __riscv_vfmul_vf_f32m8(vy_curr_epi, scale, vl_curr);
    
    // Store penultimate chunk (Stage 3)
    __riscv_vse32_v_f32m8(output, vy_prev, vl_prev);
    output += vl_prev;
    
    // Store final chunk (Stage 3)
    __riscv_vse32_v_f32m8(output, vy_curr_epi, vl_curr);
    
  } else {
    // Single-Sequence Fallback: Handles small batches where only one chunk exists.
    vint16m4_t vhx = __riscv_vwsub_vx_i16m4(vx_curr, zero_point, vl_curr);
    vfloat32m8_t vy_curr = __riscv_vfwcvt_f_x_v_f32m8(vhx, vl_curr);
    vy_curr = __riscv_vfmul_vf_f32m8(vy_curr, scale, vl_curr);
    
    __riscv_vse32_v_f32m8(output, vy_curr, vl_curr);
  }
}
