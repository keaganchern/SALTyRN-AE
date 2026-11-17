#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>

// Strategy 4: Simplify or remove unnecessary code
void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  const float slope = params->scalar.slope;
  
  // Use LMUL=8 to maximize chime length and hide pipeline latencies
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  size_t i = 0;

  // Hot loop unrolled by 2 with multiple independent pointers
  // This unrolling allows Saturn's decoupled sequencers (VLS, VXS, VSS) to overlap
  // effectively across independent element groups.
  if (i + 2 * vlmax <= n) {
    const float* in_ptr0 = input + i;
    const float* in_ptr1 = input + i + vlmax;
    float* out_ptr0 = output + i;
    float* out_ptr1 = output + i + vlmax;
    const size_t total_stride = 2 * vlmax;

    for (; i + total_stride <= n; i += total_stride) {
      // Load and generate mask for the first block
      // Mask logic: vmslt with 0 on reinterpreted i32 identifies negative floats (sign bit set)
      vfloat32m8_t vx1 = __riscv_vle32_v_f32m8(in_ptr0, vlmax);
      vbool4_t mask1 = __riscv_vmslt_vx_i32m8_b4(__riscv_vreinterpret_v_f32m8_i32m8(vx1), 0, vlmax);

      // Load for the second block (independent Load sequencer can overlap with previous VXS/VSS)
      vfloat32m8_t vx2 = __riscv_vle32_v_f32m8(in_ptr1, vlmax);
      
      // Compute masked multiply for the first block
      // Policy _mu ensures elements where mask is false (positives) remain undisturbed (identity)
      vx1 = __riscv_vfmul_vf_f32m8_mu(mask1, vx1, vx1, slope, vlmax);
      
      // Generate mask for the second block
      vbool4_t mask2 = __riscv_vmslt_vx_i32m8_b4(__riscv_vreinterpret_v_f32m8_i32m8(vx2), 0, vlmax);

      // Store first block (VSS sequencer chains with VXS)
      __riscv_vse32_v_f32m8(out_ptr0, vx1, vlmax);
      
      // Compute masked multiply for the second block
      vx2 = __riscv_vfmul_vf_f32m8_mu(mask2, vx2, vx2, slope, vlmax);

      // Store second block
      __riscv_vse32_v_f32m8(out_ptr1, vx2, vlmax);

      // Pointer multi-streaming: updates for the next unrolled iteration
      in_ptr0 += total_stride;
      in_ptr1 += total_stride;
      out_ptr0 += total_stride;
      out_ptr1 += total_stride;
    }
  }

  // Handle any remaining elements cleanly using a unified VLA tail loop.
  // This automatically manages any remaining full `vlmax` blocks and the partial tail segment.
  // Using a while loop and vsetvl reduces code size and instruction fetch pressure.
  while (i < n) {
    size_t vl = __riscv_vsetvl_e32m8(n - i);
    
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input + i, vl);
    vbool4_t mask = __riscv_vmslt_vx_i32m8_b4(__riscv_vreinterpret_v_f32m8_i32m8(vx), 0, vl);

    vx = __riscv_vfmul_vf_f32m8_mu(mask, vx, vx, slope, vl);

    __riscv_vse32_v_f32m8(output + i, vx, vl);
    
    i += vl;
  }
}
