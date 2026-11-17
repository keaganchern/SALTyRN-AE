void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float slope = params->scalar.slope;
  size_t n = batch / sizeof(float);

  while (n > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(n);

    const vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);
    const vint32m8_t vx_i = __riscv_vreinterpret_v_f32m8_i32m8(vx);
    const vbool4_t vmask = __riscv_vmslt_vx_i32m8_b4(vx_i, 0, vl);

    const vfloat32m8_t vacc = __riscv_vfmul_vf_f32m8_mu(vmask, vx, vx, slope, vl);

    __riscv_vse32_v_f32m8(output, vacc, vl);

    input += vl;
    output += vl;
    n -= vl;
  }
}