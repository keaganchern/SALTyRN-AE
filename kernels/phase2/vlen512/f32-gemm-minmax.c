// Column tiles of nr = 4v (vsetvlmax_e32m4 = 64 at VLEN=512). Weights per tile: bias[nr], then kc rows of
// w[nr], w advancing by full nr regardless of vl (tail tile zero-padded).
#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float min_v = params->scalar.min;
  const float max_v = params->scalar.max;
  const size_t nr = __riscv_vsetvlmax_e32m4();

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if (__builtin_expect(mr < 2, 0)) { a1 = a0; c1 = c0; }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if (__builtin_expect(mr <= 2, 0)) { a2 = a1; c2 = c1; }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if (__builtin_expect(mr != 4, 0)) { a3 = a2; c3 = c2; }

  size_t vl = nr;
  do {
    if (__builtin_expect(nc < nr, 0)) {
      vl = __riscv_vsetvl_e32m4(nc);
    }

    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(w, vl); w += nr;
    vfloat32m4_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;

    const float *ak0 = a0, *ak1 = a1, *ak2 = a2, *ak3 = a3;
    size_t k = kc;
    while (__builtin_expect(k >= 4 * sizeof(float), 1)) {
      vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(w, vl);
      float av0_0 = ak0[0], av1_0 = ak1[0], av2_0 = ak2[0], av3_0 = ak3[0];
      vfloat32m4_t vb1 = __riscv_vle32_v_f32m4(w + nr, vl);
      float av0_1 = ak0[1], av1_1 = ak1[1], av2_1 = ak2[1], av3_1 = ak3[1];

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, av0_0, vb0, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, av1_0, vb0, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, av2_0, vb0, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, av3_0, vb0, vl);

      vfloat32m4_t vb2 = __riscv_vle32_v_f32m4(w + 2 * nr, vl);
      float av0_2 = ak0[2], av1_2 = ak1[2], av2_2 = ak2[2], av3_2 = ak3[2];

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, av0_1, vb1, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, av1_1, vb1, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, av2_1, vb1, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, av3_1, vb1, vl);

      vfloat32m4_t vb3 = __riscv_vle32_v_f32m4(w + 3 * nr, vl);
      float av0_3 = ak0[3], av1_3 = ak1[3], av2_3 = ak2[3], av3_3 = ak3[3];

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, av0_2, vb2, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, av1_2, vb2, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, av2_2, vb2, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, av3_2, vb2, vl);

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, av0_3, vb3, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, av1_3, vb3, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, av2_3, vb3, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, av3_3, vb3, vl);

      w += 4 * nr;
      ak0 += 4; ak1 += 4; ak2 += 4; ak3 += 4;
      k -= 4 * sizeof(float);
    }

    while (__builtin_expect(k > 0, 0)) {
      vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl); w += nr;
      float av0 = *ak0++, av1 = *ak1++, av2 = *ak2++, av3 = *ak3++;
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, av0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, av1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, av2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, av3, vb, vl);
      k -= sizeof(float);
    }

    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, max_v, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, max_v, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, max_v, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, max_v, vl);

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, min_v, vl);
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, min_v, vl);
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, min_v, vl);
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, min_v, vl);
    __riscv_vse32_v_f32m4(c3, vacc3, vl);

    nc -= vl;
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
  } while (__builtin_expect(nc != 0, 0));
}
