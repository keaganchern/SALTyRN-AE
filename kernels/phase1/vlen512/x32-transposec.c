void test_rvv(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  for (size_t w = 0; w < block_width; w++) {
    const uint32_t* in_ptr = input + w;
    uint32_t* out_ptr = (uint32_t*) ((uintptr_t) output + w * output_stride);

    size_t bh = block_height;
    while (bh > 0) {
      size_t vl = __riscv_vsetvl_e32m8(bh);
      vuint32m8_t v = __riscv_vlse32_v_u32m8(in_ptr, (ptrdiff_t) input_stride, vl);
      __riscv_vse32_v_u32m8(out_ptr, v, vl);
      in_ptr = (const uint32_t*) ((uintptr_t) in_ptr + vl * input_stride);
      out_ptr += vl;
      bh -= vl;
    }
  }
}