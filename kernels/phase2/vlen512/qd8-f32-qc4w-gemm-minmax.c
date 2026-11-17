// Column tiles of nr = 4v (vsetvlmax_e32m4 = 64 at VLEN=512). Weights per tile: i32 ksum[nr], (kc/2) rows of
// nr nibble-pair bytes, f32 scale[nr], f32 bias[nr].
#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  float* c0 = c;

  /* Weights are padded to even kc in this micro-kernel. */
  kc = (kc + 1) & ~((size_t)1);

  const size_t nr = __riscv_vsetvlmax_e32m4();
  /* Per nr-column tile: nr i32 ksums + (kc/2)*nr nibble bytes
   * + nr f32 scales + nr f32 biases. */
  const size_t bytes_per_tile = nr * 12 + (kc / 2) * nr;
  const int32_t zp0 = quantization_params[0].zero_point;
  const float inv_scale0 = quantization_params[0].inv_scale;
  const float min_val = params->scalar.min;
  const float max_val = params->scalar.max;

  while (nc >= 2 * nr) {
    const uint8_t* w0_base = (const uint8_t*)w;
    const uint8_t* w1_base = w0_base + bytes_per_tile;

    size_t vl = __riscv_vsetvl_e32m4(nr);

    vint32m4_t vacc0 = __riscv_vmul_vx_i32m4(__riscv_vle32_v_i32m4((const int32_t*)w0_base, vl), zp0, vl);
    vint32m4_t vacc1 = __riscv_vmul_vx_i32m4(__riscv_vle32_v_i32m4((const int32_t*)w1_base, vl), zp0, vl);

    const int8_t* ak = a0;
    const int8_t* w0_w = (const int8_t*)(w0_base + nr * 4);
    const int8_t* w1_w = (const int8_t*)(w1_base + nr * 4);

    vint8m1_t wb0 = __riscv_vle8_v_i8m1(w0_w, vl);
    vint8m1_t wb1 = __riscv_vle8_v_i8m1(w1_w, vl);
    w0_w += nr; w1_w += nr;

    for (size_t k = 0; k < kc; k += 2) {
      const int8_t a_v0 = ak[0];
      const int8_t a_v1 = ak[1];
      ak += 2;

      vint8m1_t low0  = __riscv_vsll_vx_i8m1(wb0, 4, vl);
      vint8m1_t low1  = __riscv_vsll_vx_i8m1(wb1, 4, vl);
      vint8m1_t high0 = __riscv_vand_vx_i8m1(wb0, (int8_t)0xF0, vl);
      vint8m1_t high1 = __riscv_vand_vx_i8m1(wb1, (int8_t)0xF0, vl);

      if (k + 2 < kc) {
        wb0 = __riscv_vle8_v_i8m1(w0_w, vl);
        wb1 = __riscv_vle8_v_i8m1(w1_w, vl);
        w0_w += nr; w1_w += nr;
      }

      vint16m2_t p0_0 = __riscv_vwmul_vx_i16m2(low0, a_v0, vl);
      vint16m2_t p1_0 = __riscv_vwmul_vx_i16m2(low1, a_v0, vl);
      vint16m2_t p0_1 = __riscv_vwmul_vx_i16m2(high0, a_v1, vl);
      vint16m2_t p1_1 = __riscv_vwmul_vx_i16m2(high1, a_v1, vl);

      vacc0 = __riscv_vwadd_wv_i32m4(vacc0, p0_0, vl);
      vacc1 = __riscv_vwadd_wv_i32m4(vacc1, p1_0, vl);
      vacc0 = __riscv_vwadd_wv_i32m4(vacc0, p0_1, vl);
      vacc1 = __riscv_vwadd_wv_i32m4(vacc1, p1_1, vl);
    }

    vacc0 = __riscv_vsra_vx_i32m4(vacc0, 4, vl);
    vacc1 = __riscv_vsra_vx_i32m4(vacc1, 4, vl);

    vfloat32m4_t vout0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfloat32m4_t vout1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);

    vout0 = __riscv_vfmul_vf_f32m4(vout0, inv_scale0, vl);
    vout1 = __riscv_vfmul_vf_f32m4(vout1, inv_scale0, vl);

    const float* w0_sb = (const float*)(w0_base + nr * 4 + (kc / 2) * nr);
    const float* w1_sb = (const float*)(w1_base + nr * 4 + (kc / 2) * nr);

    vout0 = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(w0_sb + nr, vl), vout0, __riscv_vle32_v_f32m4(w0_sb, vl), vl);
    vout1 = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(w1_sb + nr, vl), vout1, __riscv_vle32_v_f32m4(w1_sb, vl), vl);

    vout0 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vout0, min_val, vl), max_val, vl);
    vout1 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vout1, min_val, vl), max_val, vl);

    __riscv_vse32_v_f32m4(c0, vout0, vl);
    float* c1 = (float*)((uintptr_t)c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vout1, vl);

    c0 = (float*)((uintptr_t)c1 + cn_stride);
    w = (const void*)(w1_base + bytes_per_tile);
    nc -= 2 * nr;
  }

  while (nc > 0) {
    size_t vl = __riscv_vsetvl_e32m4(nc < nr ? nc : nr);
    const uint8_t* w_base = (const uint8_t*)w;

    vint32m4_t vacc = __riscv_vmul_vx_i32m4(__riscv_vle32_v_i32m4((const int32_t*)w_base, vl), zp0, vl);

    const int8_t* ak = a0;
    const int8_t* w_w = (const int8_t*)(w_base + nr * 4);
    vint8m1_t wb = __riscv_vle8_v_i8m1(w_w, vl);
    w_w += nr;

    for (size_t k = 0; k < kc; k += 2) {
      const int8_t a_v0 = ak[0];
      const int8_t a_v1 = ak[1];
      ak += 2;

      vint8m1_t low  = __riscv_vsll_vx_i8m1(wb, 4, vl);
      vint8m1_t high = __riscv_vand_vx_i8m1(wb, (int8_t)0xF0, vl);

      if (k + 2 < kc) {
        wb = __riscv_vle8_v_i8m1(w_w, vl);
        w_w += nr;
      }

      vacc = __riscv_vwadd_wv_i32m4(vacc, __riscv_vwmul_vx_i16m2(low, a_v0, vl), vl);
      vacc = __riscv_vwadd_wv_i32m4(vacc, __riscv_vwmul_vx_i16m2(high, a_v1, vl), vl);
    }

    vacc = __riscv_vsra_vx_i32m4(vacc, 4, vl);
    vfloat32m4_t vout = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(vacc, vl), inv_scale0, vl);

    const float* w_sb = (const float*)(w_base + nr * 4 + (kc / 2) * nr);

    vout = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(w_sb + nr, vl), vout, __riscv_vle32_v_f32m4(w_sb, vl), vl);
    vout = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(vout, min_val, vl), max_val, vl);

    __riscv_vse32_v_f32m4(c0, vout, vl);
    c0 = (float*)((uintptr_t)c0 + cn_stride);
    w = (const void*)(w_base + bytes_per_tile);
    nc -= vl;
  }
}
