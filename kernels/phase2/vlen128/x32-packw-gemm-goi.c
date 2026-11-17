#include <riscv_vector.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#ifndef XNN_LIKELY
#define XNN_LIKELY(x) (x)
#endif

#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(x) (x)
#endif

void test_rvv(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* __restrict weights,
  const uint32_t* __restrict bias,
  const void* __restrict scale,
  uint32_t* __restrict packed_weights,
  size_t extra_bytes,
  const void* __restrict params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 2);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  do {
    // NC main loop processing 2 output channels at a time
    const uint32_t* w0 = weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      // Process bias or zero-padding. Using vectors avoids scalar store-buffer stalls.
      if XNN_LIKELY(bias != NULL) {
        vuint32m1_t vb0 = __riscv_vle32_v_u32m1(bias, 2); bias += 2;
        __riscv_vse32_v_u32m1(packed_weights, vb0, 2); packed_weights += 2;
      } else {
        vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 2);
        __riscv_vse32_v_u32m1(packed_weights, vzero, 2); packed_weights += 2;
      }

      const uint32_t* w1 = w0 + kc;
      size_t k = kc;
      size_t vlmax = __riscv_vsetvlmax_e32m4();

      // Strategy: Unroll by 2 with fixed VL to balance VLS and VSS issue queues.
      // With __restrict, the scalar core can calculate the w0/w1/packed_weights 
      // increments and loop branches far in advance of the VSS sequencer.
      while (k >= 2 * vlmax) {
        vuint32m4_t vw0_0 = __riscv_vle32_v_u32m4(w0, vlmax);
        vuint32m4_t vw1_0 = __riscv_vle32_v_u32m4(w1, vlmax);
        vuint32m4_t vw0_1 = __riscv_vle32_v_u32m4(w0 + vlmax, vlmax);
        vuint32m4_t vw1_1 = __riscv_vle32_v_u32m4(w1 + vlmax, vlmax);

        vuint32m4x2_t vw_0 = __riscv_vundefined_u32m4x2();
        vw_0 = __riscv_vset_v_u32m4_u32m4x2(vw_0, 0, vw0_0);
        vw_0 = __riscv_vset_v_u32m4_u32m4x2(vw_0, 1, vw1_0);

        vuint32m4x2_t vw_1 = __riscv_vundefined_u32m4x2();
        vw_1 = __riscv_vset_v_u32m4_u32m4x2(vw_1, 0, vw0_1);
        vw_1 = __riscv_vset_v_u32m4_u32m4x2(vw_1, 1, vw1_1);

        __riscv_vsseg2e32_v_u32m4x2(packed_weights, vw_0, vlmax);
        __riscv_vsseg2e32_v_u32m4x2(packed_weights + 2 * vlmax, vw_1, vlmax);
        
        w0 += 2 * vlmax;
        w1 += 2 * vlmax;
        packed_weights += 4 * vlmax;
        k -= 2 * vlmax;
      }

      // Standard VLA tail loop for remaining K elements
      while (k > 0) {
        size_t vl = __riscv_vsetvl_e32m4(k);
        vuint32m4_t vw0_t = __riscv_vle32_v_u32m4(w0, vl);
        vuint32m4_t vw1_t = __riscv_vle32_v_u32m4(w1, vl);

        vuint32m4x2_t vw_t = __riscv_vundefined_u32m4x2();
        vw_t = __riscv_vset_v_u32m4_u32m4x2(vw_t, 0, vw0_t);
        vw_t = __riscv_vset_v_u32m4_u32m4x2(vw_t, 1, vw1_t);
        
        __riscv_vsseg2e32_v_u32m4x2(packed_weights, vw_t, vl);
        
        w0 += vl;
        w1 += vl;
        packed_weights += 2 * vl;
        k -= vl;
      }

      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w1; // Move to next pair of input weight streams
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder of 1 channel
      if XNN_LIKELY(bias != NULL) {
        packed_weights[0] = *bias++;
        packed_weights[1] = 0;
      } else {
        vuint32m1_t vzero_bias = __riscv_vmv_v_x_u32m1(0, 2);
        __riscv_vse32_v_u32m1(packed_weights, vzero_bias, 2);
      }
      packed_weights += 2;
      
      size_t k = kc;
      size_t vlmax = __riscv_vsetvlmax_e32m4();
      vuint32m4_t vzero_k = __riscv_vmv_v_x_u32m4(0, vlmax);

      while (k >= 2 * vlmax) {
        vuint32m4_t vw0_0 = __riscv_vle32_v_u32m4(w0, vlmax);
        vuint32m4_t vw0_1 = __riscv_vle32_v_u32m4(w0 + vlmax, vlmax);

        vuint32m4x2_t vw_0 = __riscv_vundefined_u32m4x2();
        vw_0 = __riscv_vset_v_u32m4_u32m4x2(vw_0, 0, vw0_0);
        vw_0 = __riscv_vset_v_u32m4_u32m4x2(vw_0, 1, vzero_k);

        vuint32m4x2_t vw_1 = __riscv_vundefined_u32m4x2();
        vw_1 = __riscv_vset_v_u32m4_u32m4x2(vw_1, 0, vw0_1);
        vw_1 = __riscv_vset_v_u32m4_u32m4x2(vw_1, 1, vzero_k);

        __riscv_vsseg2e32_v_u32m4x2(packed_weights, vw_0, vlmax);
        __riscv_vsseg2e32_v_u32m4x2(packed_weights + 2 * vlmax, vw_1, vlmax);

        w0 += 2 * vlmax;
        packed_weights += 4 * vlmax;
        k -= 2 * vlmax;
      }

      while (k > 0) {
        size_t vl = __riscv_vsetvl_e32m4(k);
        vuint32m4_t vw0_t = __riscv_vle32_v_u32m4(w0, vl);

        vuint32m4x2_t vw_t = __riscv_vundefined_u32m4x2();
        vw_t = __riscv_vset_v_u32m4_u32m4x2(vw_t, 0, vw0_t);
        vw_t = __riscv_vset_v_u32m4_u32m4x2(vw_t, 1, vzero_k);

        __riscv_vsseg2e32_v_u32m4x2(packed_weights, vw_t, vl);

        w0 += vl;
        packed_weights += 2 * vl;
        k -= vl;
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
