#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(x) (x)
#endif
#ifndef XNN_LIKELY
#define XNN_LIKELY(x) (x)
#endif
#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(x) (x)
#endif

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

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    size_t vl = __riscv_vsetvl_e32m2(nc);

    vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(w, vl);
    w += 8;
    vfloat32m2_t vacc1 = vacc0;
    vfloat32m2_t vacc2 = vacc0;
    vfloat32m2_t vacc3 = vacc0;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float va0_0 = a0[0];
      const float va0_1 = a0[1];
      a0 += 2;
      const float va1_0 = a1[0];
      const float va1_1 = a1[1];
      a1 += 2;
      const float va2_0 = a2[0];
      const float va2_1 = a2[1];
      a2 += 2;
      const float va3_0 = a3[0];
      const float va3_1 = a3[1];
      a3 += 2;

      vfloat32m2_t vb0 = __riscv_vle32_v_f32m2(w, vl);
      vfloat32m2_t vb1 = __riscv_vle32_v_f32m2(w + 8, vl);
      w += 16;

      vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, va0_0, vb0, vl);
      vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, va1_0, vb0, vl);
      vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, va2_0, vb0, vl);
      vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, va3_0, vb0, vl);

      vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, va0_1, vb1, vl);
      vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, va1_1, vb1, vl);
      vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, va2_1, vb1, vl);
      vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, va3_1, vb1, vl);
    }
    if XNN_UNLIKELY(k != 0) {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      vfloat32m2_t vb = __riscv_vle32_v_f32m2(w, vl);
      w += 8;

      vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, va3, vb, vl);
    }

    const vbool16_t mask0 = __riscv_vmfeq_vv_f32m2_b16(vacc0, vacc0, vl);
    const vbool16_t mask1 = __riscv_vmfeq_vv_f32m2_b16(vacc1, vacc1, vl);
    const vbool16_t mask2 = __riscv_vmfeq_vv_f32m2_b16(vacc2, vacc2, vl);
    const vbool16_t mask3 = __riscv_vmfeq_vv_f32m2_b16(vacc3, vacc3, vl);

    vacc0 = __riscv_vfmin_vf_f32m2_mu(mask0, vacc0, vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m2_mu(mask1, vacc1, vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m2_mu(mask2, vacc2, vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m2_mu(mask3, vacc3, vacc3, vmax, vl);

    vacc0 = __riscv_vfmax_vf_f32m2_mu(mask0, vacc0, vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m2_mu(mask1, vacc1, vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m2_mu(mask2, vacc2, vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m2_mu(mask3, vacc3, vacc3, vmin, vl);

    __riscv_vse32_v_f32m2(c0, vacc0, vl);
    __riscv_vse32_v_f32m2(c1, vacc1, vl);
    __riscv_vse32_v_f32m2(c2, vacc2, vl);
    __riscv_vse32_v_f32m2(c3, vacc3, vl);

    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);

    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);

    nc -= vl;
  } while (nc != 0);
}