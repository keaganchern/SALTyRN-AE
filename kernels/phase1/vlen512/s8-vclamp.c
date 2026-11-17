#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int8_t output_min = (int8_t) params->scalar.min;
  const int8_t output_max = (int8_t) params->scalar.max;

  while (batch > 0) {
    const size_t vl = __riscv_vsetvl_e8m8(batch);
    vint8m8_t vacc = __riscv_vle8_v_i8m8(input, vl);
    vacc = __riscv_vmax_vx_i8m8(vacc, output_min, vl);
    vacc = __riscv_vmin_vx_i8m8(vacc, output_max, vl);
    __riscv_vse8_v_i8m8(output, vacc, vl);

    input += vl;
    output += vl;
    batch -= vl;
  }
}