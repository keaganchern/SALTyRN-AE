void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float magic = 8388608.0f;  // 0x4B000000 (2^23)

  size_t n = batch / sizeof(float);
  while (n > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(n);

    const vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    const vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx, vl);
    const vbool4_t mask = __riscv_vmfgt_vf_f32m8_b4(vabsx, magic, vl);

    vfloat32m8_t vrndabsx = __riscv_vfadd_vf_f32m8(vabsx, magic, vl);
    vrndabsx = __riscv_vfsub_vf_f32m8(vrndabsx, magic, vl);

    const vfloat32m8_t vrndabsx_signed = __riscv_vfsgnj_vv_f32m8(vrndabsx, vx, vl);
    const vfloat32m8_t vy = __riscv_vmerge_vvm_f32m8(vrndabsx_signed, vx, mask, vl);

    __riscv_vse32_v_f32m8(output, vy, vl);

    input += vl;
    output += vl;
    n -= vl;
  }
}