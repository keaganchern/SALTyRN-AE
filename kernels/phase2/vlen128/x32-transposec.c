#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

/**
 * @brief Optimized block transposition for Saturn RVV using Strategy 17.
 * 
 * Strategy 17 focuses on pre-calculating multi-row pointer offsets. By calculating
 * row-distance offsets (s1 to s8) outside the critical loop, we reduce the number
 * of scalar pointer updates per iteration. This minimizes Read-After-Write (RAW) 
 * dependency chains in the scalar core, allowing the decoupled scalar pipeline to 
 * dispatch vector instructions faster and stay ahead of the vector execution.
 */
void test_rvv(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) 
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  // Pre-calculate multi-row pointer offsets to minimize scalar arithmetic 
  // and register pressure in the inner loop (Strategy 17).
  const size_t s1 = input_stride;
  const size_t s2 = s1 * 2;
  const size_t s3 = s1 * 3;
  const size_t s4 = s1 * 4;
  const size_t s5 = s1 * 5;
  const size_t s6 = s1 * 6;
  const size_t s7 = s1 * 7;
  const size_t s8 = s1 * 8;

  size_t c = 0;
  while (block_width > 0) {
    // SEW=32, LMUL=2. Saturn's chime length (LMUL * VLEN / DLEN) at LMUL=2 
    // helps hide pipeline latencies for memory operations.
    size_t vl = __riscv_vsetvl_e32m2(block_width);
    
    // Pointer to the start of the current column block in the input.
    const uint8_t* i_ptr = (const uint8_t*)input + c * sizeof(uint32_t);
    // Pointer to the start of the current row block in the output.
    uint32_t* o_ptr = (uint32_t*)((uint8_t*)output + c * output_stride);
    
    size_t r = 0;
    if (block_height >= 4) {
      // Main loop unrolled by 8 rows. We use a single base pointer 'i_ptr' 
      // with pre-calculated offsets to maximize scalar core run-ahead.
      for (; r + 8 <= block_height; r += 8) {
        // Issue 8 unit-stride loads (VLS sequencer).
        vuint32m2_t v0 = __riscv_vle32_v_u32m2((const uint32_t*)i_ptr, vl);
        vuint32m2_t v1 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s1), vl);
        vuint32m2_t v2 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s2), vl);
        vuint32m2_t v3 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s3), vl);
        vuint32m2_t v4 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s4), vl);
        vuint32m2_t v5 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s5), vl);
        vuint32m2_t v6 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s6), vl);
        vuint32m2_t v7 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s7), vl);

        // Segmented Strided Stores: SSB handles AOS->SOA transposition at high efficiency.
        // NF=4, LMUL=2 (NF * LMUL = 8, the architectural maximum).
        vuint32m2x4_t vt0 = __riscv_vcreate_v_u32m2x4(v0, v1, v2, v3);
        __riscv_vssseg4e32_v_u32m2x4(o_ptr, (ptrdiff_t)output_stride, vt0, vl);
        
        vuint32m2x4_t vt1 = __riscv_vcreate_v_u32m2x4(v4, v5, v6, v7);
        __riscv_vssseg4e32_v_u32m2x4(o_ptr + 4, (ptrdiff_t)output_stride, vt1, vl);

        // Single scalar increment for base pointers reduces dependency overhead.
        i_ptr += s8;
        o_ptr += 8;
      }

      // Handle remaining blocks of 4 rows using the same offset logic.
      if (r + 4 <= block_height) {
        vuint32m2_t v0 = __riscv_vle32_v_u32m2((const uint32_t*)i_ptr, vl);
        vuint32m2_t v1 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s1), vl);
        vuint32m2_t v2 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s2), vl);
        vuint32m2_t v3 = __riscv_vle32_v_u32m2((const uint32_t*)(i_ptr + s3), vl);
        
        vuint32m2x4_t vt = __riscv_vcreate_v_u32m2x4(v0, v1, v2, v3);
        __riscv_vssseg4e32_v_u32m2x4(o_ptr, (ptrdiff_t)output_stride, vt, vl);
        
        i_ptr += s4;
        o_ptr += 4;
        r += 4;
      }
    }
    
    // Tail cleanup for remaining block_height % 4 rows.
    // While vsse is 1 element/cycle, it is necessary for non-multiple-of-4 row counts.
    for (; r < block_height; ++r) {
      vuint32m2_t v = __riscv_vle32_v_u32m2((const uint32_t*)i_ptr, vl);
      __riscv_vsse32_v_u32m2(o_ptr, (ptrdiff_t)output_stride, v, vl);
      i_ptr += s1;
      o_ptr += 1;
    }
    
    c += vl;
    block_width -= vl;
  }
}

