void test_rvv(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  size_t vlmax = __riscv_vsetvlmax_e32m8();
  vuint32m8_t vacc = __riscv_vmv_v_x_u32m8(0, vlmax);

  while (batch > 0) {
    size_t vl = __riscv_vsetvl_e8m2(batch);
    vuint8m2_t vt = __riscv_vle8_v_u8m2(input, vl);
    vuint32m8_t vt32 = __riscv_vzext_vf4_u32m8(vt, vl);
    vacc = __riscv_vadd_vv_u32m8_tu(vacc, vacc, vt32, vl);
    input += vl;
    batch -= vl;
  }

  vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 1);
  vuint32m1_t vres = __riscv_vredsum_vs_u32m8_u32m1(vacc, vzero, vlmax);
  *output += __riscv_vmv_x_s_u32m1_u32(vres);
}