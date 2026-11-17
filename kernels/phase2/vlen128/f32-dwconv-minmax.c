#include <assert.h>
#include <stdint.h>
#include <riscv_vector.h>

struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

/*
 * pack_weights: Converts the canonical LMUL=8 blocked layout (from the reference)
 * into an LMUL=4 blocked layout optimized for register hoisting on Saturn.
 * Saturn's short-vector pipeline (DLEN=VLEN or similar) benefits from 
 * maximizing LMUL to hide FMA latency (4 cycles). LMUL=4 provides enough 
 * registers (8 groups total) to hoist all weights for this 3-tap kernel.
 */
void pack_weights(const void* src, size_t channels, void* dst, size_t dst_cap) {
  const float* s = (const float*)src;
  float* d = (float*)dst;
  
  for (size_t c = 0; c < channels; ) {
    // Identify the block size used by the reference (LMUL=8)
    size_t vl8 = __riscv_vsetvl_e32m8(channels - c);
    const float* b8 = s;
    const float* w0_8 = s + vl8;
    const float* w1_8 = s + 2 * vl8;
    const float* w2_8 = s + 3 * vl8;
    s += 4 * vl8;
    
    // Sub-divide the LMUL=8 block into LMUL=4 blocks for our kernel
    size_t processed = 0;
    while (processed < vl8) {
      size_t vl4 = __riscv_vsetvl_e32m4(vl8 - processed);
      for (size_t i = 0; i < vl4; i++) *d++ = b8[processed + i];
      for (size_t i = 0; i < vl4; i++) *d++ = w0_8[processed + i];
      for (size_t i = 0; i < vl4; i++) *d++ = w1_8[processed + i];
      for (size_t i = 0; i < vl4; i++) *d++ = w2_8[processed + i];
      processed += vl4;
    }
    c += vl8;
  }
}

void test_rvv(
    size_t channels,
    size_t output_width,
    const float** restrict input,
    const float* restrict weights,
    float* restrict output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* restrict zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);

  const float min_clamp = params->scalar.min;
  const float max_clamp = params->scalar.max;
  const size_t output_stride_bytes = channels * sizeof(float) + output_increment;
  const float* weights_ptr = weights;

  for (size_t c_idx = 0; c_idx < channels; ) {
    const size_t vl = __riscv_vsetvl_e32m4(channels - c_idx);
    
    // Strategy: Register Hoisting
    // At LMUL=4, we use 4 vector register groups for weights (16 registers total).
    // This removes 4 VLS instructions from the inner spatial loop.
    vfloat32m4_t vbias = __riscv_vle32_v_f32m4(weights_ptr, vl);
    vfloat32m4_t vw0   = __riscv_vle32_v_f32m4(weights_ptr + vl, vl);
    vfloat32m4_t vw1   = __riscv_vle32_v_f32m4(weights_ptr + 2 * vl, vl);
    vfloat32m4_t vw2   = __riscv_vle32_v_f32m4(weights_ptr + 3 * vl, vl);
    weights_ptr += 4 * vl;

    const float** current_input = input;
    float* current_output = output;
    size_t current_input_offset = input_offset;

    // Strategy: Software Pipelining
    // Initial load for the first pixel (ow = 0) to prime the pipeline.
    const float* i0 = current_input[0];
    const float* i1 = current_input[1];
    const float* i2 = current_input[2];
    if (i0 != zero) i0 = (const float*) ((uintptr_t) i0 + current_input_offset);
    if (i1 != zero) i1 = (const float*) ((uintptr_t) i1 + current_input_offset);
    if (i2 != zero) i2 = (const float*) ((uintptr_t) i2 + current_input_offset);

    vfloat32m4_t vi0 = __riscv_vle32_v_f32m4(i0 + c_idx, vl);
    vfloat32m4_t vi1 = __riscv_vle32_v_f32m4(i1 + c_idx, vl);
    vfloat32m4_t vi2 = __riscv_vle32_v_f32m4(i2 + c_idx, vl);

    for (size_t ow = 0; ow < output_width; ow++) {
      // Initialize accumulator with hoisted bias (Vector-to-Vector copy, VXS sequencer)
      vfloat32m4_t vacc = vbias;

      // Scalar overlap: Prepare address calculations for next pixel (ow + 1)
      const float** next_input = (const float**) ((uintptr_t) current_input + input_stride);
      size_t next_offset = current_input_offset + input_pixel_stride;

      /*
       * Strategy: Overlap Compute (VXS) and Data Movement (VLS)
       * We interleave vfmacc (Arithmetic) with vle (Load for next iteration).
       * Saturn's decoupled sequencers allow the VLS unit to pre-load next 
       * pixel data while the VXS unit processes the current one.
       */
      vacc = __riscv_vfmacc_vv_f32m4(vacc, vi0, vw0, vl);
      if (ow + 1 < output_width) {
        const float* ni0 = next_input[0];
        if (ni0 != zero) ni0 = (const float*) ((uintptr_t) ni0 + next_offset);
        vi0 = __riscv_vle32_v_f32m4(ni0 + c_idx, vl);
      }

      vfloat32m4_t vacc1 = __riscv_vfmul_vv_f32m4(vi1, vw1, vl);   // second chain, summed at the end, as Neon does
      if (ow + 1 < output_width) {
        const float* ni1 = next_input[1];
        if (ni1 != zero) ni1 = (const float*) ((uintptr_t) ni1 + next_offset);
        vi1 = __riscv_vle32_v_f32m4(ni1 + c_idx, vl);
      }

      vacc = __riscv_vfmacc_vv_f32m4(vacc, vi2, vw2, vl);
      if (ow + 1 < output_width) {
        const float* ni2 = next_input[2];
        if (ni2 != zero) ni2 = (const float*) ((uintptr_t) ni2 + next_offset);
        vi2 = __riscv_vle32_v_f32m4(ni2 + c_idx, vl);
      }

      // Final compute stages (Clamping) and Store (VSS sequencer)
      // Clamping uses the FP-Convert pipe, which can overlap with FMA or Store units.
      vacc = __riscv_vfadd_vv_f32m4(vacc, vacc1, vl);
      vacc = __riscv_vfmax_vf_f32m4(vacc, min_clamp, vl);
      vacc = __riscv_vfmin_vf_f32m4(vacc, max_clamp, vl);
      __riscv_vse32_v_f32m4(current_output + c_idx, vacc, vl);

      // Advance scalar state for next spatial step
      current_input = next_input;
      current_input_offset = next_offset;
      current_output = (float*) ((uintptr_t) current_output + output_stride_bytes);
    }

    c_idx += vl;
  }
}

