#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

void test_rvv(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    size_t index_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(channels != 0);

  do {
    size_t c_total = channels;
    size_t c_base = 0;

    while (c_total > 0) {
      const size_t vl = __riscv_vsetvl_e32m8(c_total);

      // Strategy: Hoist redundant offset calculations out of the pooling loop
      // Combine the byte equivalents of input_offset and c_base into a single scalar addition.
      // input_offset is in bytes, while c_base is the element index (float).
      const size_t combined_offset = input_offset + (c_base * sizeof(float));

      // Hoist the first element initialization to set the baseline max and index.
      const float** id = input;
      const float* ptr_k0 = (const float*) ((uintptr_t) (*id++) + combined_offset);
      
      vfloat32m8_t vmax = __riscv_vle32_v_f32m8(ptr_k0, vl);
      vuint32m8_t vidx = __riscv_vmv_v_x_u32m8(0, vl);

      uint32_t k = 1;
      const uint32_t pool_limit = (uint32_t)pooling_elements;

      // Unroll by 4 to further reduce scalar pointer arithmetic and control logic overhead.
      // Re-using 'vk' and 'm' ensures we consume exactly 25 vector registers (8 for vmax, 
      // 8 for vidx, 8 for vk, and 1 for the mask m), preventing the compiler from 
      // scheduling excessive simultaneous loads that would spill to the stack at LMUL=8.
      for (; k + 3 < pool_limit; k += 4) {
        const float* ptr_k  = (const float*) ((uintptr_t) id[0] + combined_offset);
        const float* ptr_k1 = (const float*) ((uintptr_t) id[1] + combined_offset);
        const float* ptr_k2 = (const float*) ((uintptr_t) id[2] + combined_offset);
        const float* ptr_k3 = (const float*) ((uintptr_t) id[3] + combined_offset);
        id += 4;

        // Iteration k
        vfloat32m8_t vk = __riscv_vle32_v_f32m8(ptr_k, vl);
        vbool4_t m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k, m, vl);

        // Iteration k+1
        vk = __riscv_vle32_v_f32m8(ptr_k1, vl);
        m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k + 1, m, vl);

        // Iteration k+2
        vk = __riscv_vle32_v_f32m8(ptr_k2, vl);
        m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k + 2, m, vl);

        // Iteration k+3
        vk = __riscv_vle32_v_f32m8(ptr_k3, vl);
        m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k + 3, m, vl);
      }

      // Handle unrolled remainder by 2
      for (; k + 1 < pool_limit; k += 2) {
        const float* ptr_k  = (const float*) ((uintptr_t) id[0] + combined_offset);
        const float* ptr_k1 = (const float*) ((uintptr_t) id[1] + combined_offset);
        id += 2;

        vfloat32m8_t vk = __riscv_vle32_v_f32m8(ptr_k, vl);
        vbool4_t m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k, m, vl);

        vk = __riscv_vle32_v_f32m8(ptr_k1, vl);
        m = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, m, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k + 1, m, vl);
      }

      // Single-element tail loop
      for (; k < pool_limit; k++) {
        const float* ptr_kk = (const float*) ((uintptr_t) id[0] + combined_offset);
        id += 1;
        vfloat32m8_t vk = __riscv_vle32_v_f32m8(ptr_kk, vl);

        vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(vk, vmax, vl);
        vmax = __riscv_vmerge_vvm_f32m8(vmax, vk, mask, vl);   // select by the mask, as the index is
        vidx = __riscv_vmerge_vxm_u32m8(vidx, k, mask, vl);
      }

      // Store results to output and index buffers
      __riscv_vse32_v_f32m8(output + c_base, vmax, vl);
      __riscv_vse32_v_u32m8(index + c_base, vidx, vl);

      c_total -= vl;
      c_base += vl;
    }

    // Step to the next output pixel using the provided byte-stride increments.
    input = (const float**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
    index = (uint32_t*) ((uintptr_t) index + index_increment);
  } while (--output_pixels != 0);
}
