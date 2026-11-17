#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

#define XNN_OOB_READS

/**
 * Optimized RVV implementation of f32 minmax (clamping) kernel.
 * Target: Saturn RVV Microarchitecture.
 * 
 * Optimization Strategies Applied:
 * 1. Hoisted Channel Strip-Mining: vsetvl is called outside the pixel loop.
 * 2. Bound Splat Hoisting (Strategy 13): vmin and vmax are splatted once per channel chunk.
 * 3. Maximize LMUL: LMUL=8 is used to hide 4-stage FMA/Arithmetic pipeline latencies.
 * 4. Sequencer Interleaving: Memory loads (VLS) and arithmetic (VXS) are interleaved
 *    to maximize chaining opportunities in Saturn's decoupled backend.
 * 5. Optimal Register Packing: Uses all 32 vector registers (3x8 for constants/acc, 1x8 for loads).
 */
void test_rvv(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  size_t c_offset = 0;
  while (channels > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(channels);
    
    // Strategy 13: Bound Splat Hoisting
    // Splatting vmin/vmax here eliminates redundant scalar-to-vector transfers 
    // and vfmv.v.f instructions from the inner pixel loop.
    const vfloat32m8_t vmin_vec = __riscv_vfmv_v_f_f32m8(vmin, vl);
    const vfloat32m8_t vmax_vec = __riscv_vfmv_v_f_f32m8(vmax, vl);
    
    const float** ik_pixel = input;
    size_t input_offset_pixel = input_offset;
    float* output_pixel = output;
    
    // Pre-calculate the byte-offset for the current channel chunk.
    const size_t c_byte_offset = c_offset * sizeof(float);

    for (size_t p = 0; p < output_pixels; p++) {
      // Initialize accumulator with the pre-splatted minimum bound.
      vfloat32m8_t vacc = vmin_vec;
      
      const float** ik = ik_pixel;
      int k = (int) kernel_elements;
      // Combine pixel-specific and channel-specific offsets for address generation.
      size_t combined_offset = input_offset_pixel + c_byte_offset;
      
      // Unroll kernel loop to interleave VLS (Load) and VXS (Execute) instructions.
      // Saturn's sequencers allow Load and Max to chain, overlapping execution.
      while (k >= 4) {
        const float* p0 = (const float*) ((uintptr_t) ik[0] + combined_offset);
        const float* p1 = (const float*) ((uintptr_t) ik[1] + combined_offset);
        const float* p2 = (const float*) ((uintptr_t) ik[2] + combined_offset);
        const float* p3 = (const float*) ((uintptr_t) ik[3] + combined_offset);
        ik += 4;
        
        vfloat32m8_t vi0 = __riscv_vle32_v_f32m8(p0, vl);
        vacc = __riscv_vfmax_vv_f32m8(vacc, vi0, vl);
        
        vfloat32m8_t vi1 = __riscv_vle32_v_f32m8(p1, vl);
        vacc = __riscv_vfmax_vv_f32m8(vacc, vi1, vl);
        
        vfloat32m8_t vi2 = __riscv_vle32_v_f32m8(p2, vl);
        vacc = __riscv_vfmax_vv_f32m8(vacc, vi2, vl);
        
        vfloat32m8_t vi3 = __riscv_vle32_v_f32m8(p3, vl);
        vacc = __riscv_vfmax_vv_f32m8(vacc, vi3, vl);
        
        k -= 4;
      }
      
      while (k > 0) {
        const float* pk = (const float*) ((uintptr_t) ik[0] + combined_offset);
        ik += 1;
        vfloat32m8_t vik = __riscv_vle32_v_f32m8(pk, vl);
        vacc = __riscv_vfmax_vv_f32m8(vacc, vik, vl);
        k -= 1;
      }
      
      // Final clamp using the pre-splatted vmax vector. 
      // This is a vector-vector operation, keeping the VXS sequencer busy.
      vacc = __riscv_vfmin_vv_f32m8(vacc, vmax_vec, vl);
      
      // Store result slice (VSS sequencer). DAE allows the scalar core to 
      // perform the bookkeeping below while this store is in flight.
      __riscv_vse32_v_f32m8(output_pixel + c_offset, vacc, vl);
      
      // Scalar bookkeeping: Update pointers and offsets for the next pixel.
      ik_pixel = (const float**) ((uintptr_t) ik_pixel + input_increment);
      input_offset_pixel += input_pixel_stride;
      output_pixel = (float*) ((uintptr_t) output_pixel + output_increment);
    }
    
    // Strip-mining: Move to the next block of channels.
    channels -= vl;
    c_offset += vl;
  }
}
