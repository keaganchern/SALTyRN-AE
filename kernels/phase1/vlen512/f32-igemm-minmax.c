#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(x) (x)
#endif
#ifndef XNN_LIKELY
#define XNN_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif
#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

struct xnn_f32_minmax_params;

void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    const size_t vl = __riscv_vsetvl_e32m1(nc < 8 ? nc : 8);

    vfloat32m1_t vacc0 = __riscv_vle32_v_f32m1(w, vl);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float va0_0 = a0[0];
        const float va0_1 = a0[1];
        a0 += 2;

        const vfloat32m1_t vb0 = __riscv_vle32_v_f32m1(w, vl);
        w += 8;
        vacc0 = __riscv_vfmacc_vf_f32m1(vacc0, va0_0, vb0, vl);

        const vfloat32m1_t vb1 = __riscv_vle32_v_f32m1(w, vl);
        w += 8;
        vacc0 = __riscv_vfmacc_vf_f32m1(vacc0, va0_1, vb1, vl);
      }
      if XNN_UNLIKELY(k != 0) {
        const float va0 = *a0;

        const vfloat32m1_t vb = __riscv_vle32_v_f32m1(w, vl);
        w += 8;
        vacc0 = __riscv_vfmacc_vf_f32m1(vacc0, va0, vb, vl);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0 = __riscv_vfmin_vf_f32m1(vacc0, params->scalar.max, vl);
    vacc0 = __riscv_vfmax_vf_f32m1(vacc0, params->scalar.min, vl);

    __riscv_vse32_v_f32m1(c0, vacc0, vl);

    if XNN_LIKELY(nc >= 8) {
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      nc = 0;
    }
  } while (nc != 0);
}