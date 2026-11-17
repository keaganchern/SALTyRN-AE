void test_rvv(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_u8_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint8_t output_min = (uint8_t) params->scalar.min;
  const uint8_t output_max = (uint8_t) params->scalar.max;

  while (batch > 0) {
    const size_t vl = __riscv_vsetvl_e8m8(batch);
    vuint8m8_t vacc = __riscv_vle8_v_u8m8(input, vl);
    vacc = __riscv_vmaxu_vx_u8m8(vacc, output_min, vl);
    vacc = __riscv_vminu_vx_u8m8(vacc, output_max, vl);
    __riscv_vse8_v_u8m8(output, vacc, vl);
    input += vl;
    output += vl;
    batch -= vl;
  }
}