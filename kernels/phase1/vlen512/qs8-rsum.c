struct xnn_qs8_rsum_params;

void test_rvv(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const size_t max_vl = __riscv_vsetvlmax_e32m8();
  vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, max_vl);

  while (batch > 0) {
    const size_t vl = __riscv_vsetvl_e8m2(batch);
    const vint8m2_t vt = __riscv_vle8_v_i8m2(input, vl);
    const vint32m8_t vt32 = __riscv_vsext_vf4_i32m8(vt, vl);
    vacc = __riscv_vadd_vv_i32m8_tu(vacc, vacc, vt32, vl);
    input += vl;
    batch -= vl;
  }

  const vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t vres = __riscv_vredsum_vs_i32m8_i32m1(vacc, vzero, max_vl);
  const int32_t vacc_scalar = __riscv_vmv_x_s_i32m1_i32(vres);

  *output += vacc_scalar;
}