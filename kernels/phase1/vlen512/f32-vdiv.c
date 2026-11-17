void test_rvv(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  while (n > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    const vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    const vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);

    const vfloat32m8_t vacc = __riscv_vfdiv_vv_f32m8(va, vb, vl);

    __riscv_vse32_v_f32m8(output, vacc, vl);
    input_a += vl;
    input_b += vl;
    output += vl;
    n -= vl;
  }
}