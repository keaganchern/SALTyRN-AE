#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

/**
 * Saturn RVV Optimized GEMM Kernel
 * Strategy: LMUL=8 Row-Grouped Multi-Pass Execution with N-Loop Pipelining
 * 
 * 1. Maximize Chime Length: Uses LMUL=8 to maximize hardware occupancy and 
 *    hide the 4-stage FMA latency on Saturn's VXS sequencer.
 * 2. N-Loop Pipelining: Software-pipelines the weight bias loading across 
 *    column blocks. The VLS (Load) sequencer fetches the next block's bias 
 *    while the VXS (Execute) and VSS (Store) sequencers finalize the current block.
 * 3. Register-Aware Row Grouping: Processes mr=4 in two passes of 2 rows each 
 *    to stay within the 32-vector register limit at LMUL=8.
 */
void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  (void) cn_stride;

  const size_t k_floats = kc / sizeof(float);
  const float min_f = params->scalar.min;
  const float max_f = params->scalar.max;

  // Prime the pipeline with the first bias load for the initial nc block
  size_t vl_curr = __riscv_vsetvl_e32m8(nc);
  vfloat32m8_t next_bias = __riscv_vle32_v_f32m8(w, vl_curr);

  while (nc > 0) {
    const size_t vl = vl_curr;

    // Process rows in groups of 2.
    // Register breakdown (32 total):
    // - vacc0: 8 registers (v0-v7)
    // - vacc1: 8 registers (v8-v15)
    // - vb_c:  8 registers (v16-v23)
    // - next_bias: 8 registers (v24-v31)
    for (size_t r = 0; r < mr; r += 2) {
      const float* w_ptr = w;
      
      // Load/forward bias values
      vfloat32m8_t vacc0;
      if (r == 0) {
        // Reuse pre-fetched bias for the first row-group
        vacc0 = next_bias;
      } else {
        // Subsequent row-groups re-load the same bias to free next_bias's registers
        vacc0 = __riscv_vle32_v_f32m8(w_ptr, vl);
      }
      vfloat32m8_t vacc1 = vacc0;
      w_ptr += vl;

      const float* ar0 = (const float*) ((uintptr_t) a + r * a_stride);
      const float* ar1 = (r + 1 < mr) ? (const float*) ((uintptr_t) ar0 + a_stride) : ar0;

      if (k_floats > 0) {
        // Prime the k-loop weights
        float a0_c = ar0[0];
        float a1_c = (r + 1 < mr) ? ar1[0] : 0.0f;
        vfloat32m8_t vb_c = __riscv_vle32_v_f32m8(w_ptr, vl);
        w_ptr += vl;

        for (size_t k = 1; k < k_floats; ++k) {
          // Fetch next scalar activations
          float a0_n = ar0[k];
          float a1_n = (r + 1 < mr) ? ar1[k] : 0.0f;

          // Dispatch FMAs (VXS Sequencer)
          // 2 FMAs per iteration (16+ cycle chime) hide the weight load (8+ cycle chime)
          vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a0_c, vb_c, vl);
          if (r + 1 < mr) {
            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, a1_c, vb_c, vl);
          }

          // Fetch next weight block (VLS Sequencer)
          // DAE handles overlapping this load with current FMAs
          vb_c = __riscv_vle32_v_f32m8(w_ptr, vl);
          w_ptr += vl;

          a0_c = a0_n; a1_c = a1_n;
        }

        // Final FMA drain
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a0_c, vb_c, vl);
        if (r + 1 < mr) {
          vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, a1_c, vb_c, vl);
        }
      }

      // Clamping and ReLU activation (VXS Sequencer)
      vacc0 = __riscv_vfmin_vf_f32m8(vacc0, max_f, vl);
      vacc0 = __riscv_vfmax_vf_f32m8(vacc0, min_f, vl);
      if (r + 1 < mr) {
        vacc1 = __riscv_vfmin_vf_f32m8(vacc1, max_f, vl);
        vacc1 = __riscv_vfmax_vf_f32m8(vacc1, min_f, vl);
      }

      // Store results (VSS Sequencer)
      float* cr0 = (float*) ((uintptr_t) c + r * cm_stride);
      __riscv_vse32_v_f32m8(cr0, vacc0, vl);
      if (r + 1 < mr) {
        float* cr1 = (float*) ((uintptr_t) cr0 + cm_stride);
        __riscv_vse32_v_f32m8(cr1, vacc1, vl);
      }
    }

    // Step pointers to the next block of columns
    w += (k_floats + 1) * vl;
    c += vl;
    nc -= vl;

    // Prefetch logic: issue bias load for the next nc block immediately
    if (nc > 0) {
      vl_curr = __riscv_vsetvl_e32m8(nc);
      next_bias = __riscv_vle32_v_f32m8(w, vl_curr);
    }
  }
}
