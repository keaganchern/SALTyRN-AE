#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

// XNNPACK-style params structure for clamping
struct xnn_f32_minmax_params {
  struct {
    float max;
    float min;
  } scalar;
};

#define XNN_UNPREDICTABLE(exp) (__builtin_expect(!!(exp), 0))   // parenthesized: used as `if XNN_UNPREDICTABLE(c)`

/**
 * Optimized GEMM-like kernel for Saturn RVV Microarchitecture.
 * 
 * Applies Strategy 2: Use multiple accumulators to hide FMA latency and sustain issue throughput.
 */
void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  (void) cm_stride;
  (void) cn_stride;

  const size_t n_inputs = ks / sizeof(void*);
  const size_t k_elements = kc / sizeof(float);
  const float* a_ptrs[n_inputs];
  
  // Pre-calculate input row pointers once
  for (size_t i = 0; i < n_inputs; i++) {
    const float* a_row_ptr = a[i];
    if XNN_UNPREDICTABLE(a_row_ptr != zero) {
      a_row_ptr = (const float*) ((uintptr_t) a_row_ptr + a_offset);
    }
    a_ptrs[i] = a_row_ptr;
  }

  // Pre-calculate the size of a weight strip (Bias + Weights for all rows)
  const size_t strip_size = 1 + n_inputs * k_elements;
  size_t n_processed = 0;

  while (nc > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(nc);
    
    // The weight layout is assumed to be strip-major: [Strip 0][Strip 1]...
    // where each strip is strip_size * vl elements long.
    const float* w_strip = w + n_processed * strip_size;
    
    // Load bias (first block of the strip) into vacc0
    vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(w_strip, vl);
    // Initialize secondary accumulator

    for (size_t i = 0; i < n_inputs; i++) {
      const float* restrict a0 = a_ptrs[i];
      
      // Row-Level Zero Skipping
      // The zero row is multiplied like any other, as Neon does (0 * w is NaN for a non-finite w).

      // Pre-calculate total displacement for the row within the strip.
      // Weights for row i start after the bias and previous rows.
      const float* restrict w_row = w_strip + (1 + i * k_elements) * vl;

      size_t k = 0;

      // One accumulator in Neon's order, and no test of the activation against 0.0f (0 * w is NaN for a
      // non-finite w).
      for (; k + 3 < k_elements; k += 4) {
        const float a_val0 = a0[k];
        const float a_val1 = a0[k+1];
        const float a_val2 = a0[k+2];
        const float a_val3 = a0[k+3];

        const float* w_base = w_row + k * vl;
        vfloat32m8_t vb0 = __riscv_vle32_v_f32m8(w_base, vl);
        vfloat32m8_t vb1 = __riscv_vle32_v_f32m8(w_base + vl, vl);
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a_val0, vb0, vl);
        vfloat32m8_t vb2 = __riscv_vle32_v_f32m8(w_base + 2 * vl, vl);
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a_val1, vb1, vl);
        vfloat32m8_t vb3 = __riscv_vle32_v_f32m8(w_base + 3 * vl, vl);
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a_val2, vb2, vl);
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a_val3, vb3, vl);
      }

      // Remainder logic for K steps that don't fit in modulo 4
      for (; k < k_elements; k++) {
        const float a_val = a0[k];
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(w_row + k * vl, vl);
        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a_val, vb, vl);
      }
    }


    // Clamping (min/max)
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(params->scalar.min, vl);
    vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(params->scalar.max, vl);
    // Clamp to max first, then min, as Neon does (on a NaN accumulator the two orders differ).
    vacc0 = __riscv_vfmin_vv_f32m8(vacc0, vmax, vl);
    vacc0 = __riscv_vfmax_vv_f32m8(vacc0, vmin, vl);

    // Store results
    __riscv_vse32_v_f32m8(c + n_processed, vacc0, vl);
    
    nc -= vl;
    n_processed += vl;
  }
}
