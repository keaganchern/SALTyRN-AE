// Column tiles of nr = 4v (vsetvlmax_e32m4 = 64 at VLEN=512). Weights per tile: bias[nr], then per spatial
// position and k step w[nr], w advancing by full nr (tail tile zero-padded).
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

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
  assert(mr <= 1);
  assert(a != NULL && w != NULL && c != NULL);

  const size_t kc_elems = kc / sizeof(float);
  const size_t ks_groups = ks / sizeof(void*);
  const size_t nr = __riscv_vsetvlmax_e32m4();
  const size_t tile_stride = nr * (1 + ks_groups * kc_elems);

  const vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(params->scalar.max, nr);
  const vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(params->scalar.min, nr);

  /* Primary loop: 2 tiles of 4v — one indirection traversal per 2*nr cols. */
  while (__builtin_expect(nc >= 2 * nr, 1)) {
    const size_t vl = nr;
    const float* w0 = w;
    const float* w1 = w + tile_stride;

    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(w0, vl); w0 += nr;
    vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(w1, vl); w1 += nr;

    size_t p = ks_groups;
    const float** ai = a;
    do {
      const float* restrict a0 = *ai++;
      if (a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      {

        const float* a_end_4 = a0 + (kc_elems & ~((size_t)3));
        const float* a_end   = a0 + kc_elems;

        while (a0 < a_end_4) {
          const float va0 = a0[0];
          const float va1 = a0[1];
          const float va2 = a0[2];
          const float va3 = a0[3];
          a0 += 4;

          vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, __riscv_vle32_v_f32m4(w0, vl), vl);
          vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va0, __riscv_vle32_v_f32m4(w1, vl), vl);
          vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va1, __riscv_vle32_v_f32m4(w0 + nr, vl), vl);
          vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, __riscv_vle32_v_f32m4(w1 + nr, vl), vl);
          vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va2, __riscv_vle32_v_f32m4(w0 + 2 * nr, vl), vl);
          vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va2, __riscv_vle32_v_f32m4(w1 + 2 * nr, vl), vl);
          vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va3, __riscv_vle32_v_f32m4(w0 + 3 * nr, vl), vl);
          vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va3, __riscv_vle32_v_f32m4(w1 + 3 * nr, vl), vl);

          w0 += 4 * nr; w1 += 4 * nr;
        }
        while (a0 < a_end) {
          const float va = *a0++;
          vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va, __riscv_vle32_v_f32m4(w0, vl), vl);
          vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va, __riscv_vle32_v_f32m4(w1, vl), vl);
          w0 += nr; w1 += nr;
        }
      }
    } while (--p != 0);

    vacc0 = __riscv_vfmin_vv_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vv_f32m4(vacc1, vmax, vl);
    vacc0 = __riscv_vfmax_vv_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vv_f32m4(vacc1, vmin, vl);

    __riscv_vse32_v_f32m4(c, vacc0, vl);
    __riscv_vse32_v_f32m4((float*)((uintptr_t)c + cn_stride), vacc1, vl);

    c = (float*) ((uintptr_t) c + 2 * cn_stride);
    w += 2 * tile_stride;
    nc -= 2 * nr;
  }

  /* Single-tile loop, vl-tail for nc < nr. */
  while (__builtin_expect(nc > 0, 0)) {
    size_t vl = __riscv_vsetvl_e32m4(nc < nr ? nc : nr);
    const float* w0 = w;
    vfloat32m4_t vacc = __riscv_vle32_v_f32m4(w0, vl); w0 += nr;

    size_t p = ks_groups;
    const float** ai = a;
    do {
      const float* restrict a0 = *ai++;
      if (a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      {

        const float* a_end_4 = a0 + (kc_elems & ~((size_t)3));
        const float* a_end   = a0 + kc_elems;

        while (a0 < a_end_4) {
          const float va0 = a0[0];
          const float va1 = a0[1];
          const float va2 = a0[2];
          const float va3 = a0[3];
          a0 += 4;
          vacc = __riscv_vfmacc_vf_f32m4(vacc, va0, __riscv_vle32_v_f32m4(w0, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m4(vacc, va1, __riscv_vle32_v_f32m4(w0 + nr, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m4(vacc, va2, __riscv_vle32_v_f32m4(w0 + 2 * nr, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m4(vacc, va3, __riscv_vle32_v_f32m4(w0 + 3 * nr, vl), vl);
          w0 += 4 * nr;
        }
        while (a0 < a_end) {
          vacc = __riscv_vfmacc_vf_f32m4(vacc, *a0++, __riscv_vle32_v_f32m4(w0, vl), vl);
          w0 += nr;
        }
      }
    } while (--p != 0);

    vacc = __riscv_vfmin_vv_f32m4(vacc, vmax, vl);
    vacc = __riscv_vfmax_vv_f32m4(vacc, vmin, vl);
    __riscv_vse32_v_f32m4(c, vacc, vl);

    c = (float*) ((uintptr_t) c + cn_stride);
    w += tile_stride;
    nc -= vl;
  }
}
