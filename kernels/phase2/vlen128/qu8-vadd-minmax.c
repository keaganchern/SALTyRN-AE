#include <riscv_vector.h>
#include <stdint.h>
#include <assert.h>

// XNNPACK's parameter structure (src/xnnpack/microparams.h).
struct xnn_qu8_add_minmax_params {
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int32_t bias;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } scalar;
};

void test_rvv(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const struct xnn_qu8_add_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  // Hoist scalar parameters and pre-calculate the unified bias.
  const int32_t a_mult = params->scalar.a_multiplier;
  const int32_t b_mult = params->scalar.b_multiplier;
  const int32_t a_zp = (int32_t)params->scalar.a_zero_point;
  const int32_t b_zp = (int32_t)params->scalar.b_zero_point;
  const int16_t o_zp = params->scalar.output_zero_point;
  const uint32_t shift = (uint32_t)params->scalar.shift;
  const uint8_t output_min = params->scalar.output_min;
  const uint8_t output_max = params->scalar.output_max;

  // Manual RNU rounding term for the arithmetic shift stage.
  // The rounding term (1 << (shift - 1)) is added before the narrowing shift.
  const int32_t round_term = (shift > 0) ? (1 << (shift - 1)) : 0;
  // Unified bias incorporates the input zero-point offsets and rounding. The output zero point is added after the
  // shift (o_zp << shift overflows int32 for large shifts).
  const int32_t bias = round_term - (a_zp * a_mult + b_zp * b_mult);

  // Retrieve maximum VL for the base data type (e8m2/e32m8 share the same VLMAX).
  const size_t vlmax = __riscv_vsetvlmax_e8m2();
  
  // Pre-splat the unified bias into a vector register group.
  vint32m8_t v_bias = __riscv_vmv_v_x_i32m8(bias, vlmax);

  size_t i = 0;
  // --- Fixed-VL Hot Loop ---
  // Processes elements uniformly at `vlmax` length to entirely eliminate `vsetvl` overhead.
  for (; i + vlmax <= batch; i += vlmax) {
    // Vector Loads (VLS Sequencer)
    vuint8m2_t va = __riscv_vle8_v_u8m2(input_a + i, vlmax);
    vuint8m2_t vb = __riscv_vle8_v_u8m2(input_b + i, vlmax);

    // Strategy 10: Independent Promotion ILP.
    // Issue both promotions to the IntPipe/ShiftPipe back-to-back.
    // In Saturn's decoupled execution, this allows the VXS sequencer to dispatch 
    // these extension operations to their respective functional unit queues before 
    // the MulPipe becomes the bottleneck for the subsequent multiplications.
    vint32m8_t va32 = __riscv_vreinterpret_v_u32m8_i32m8(__riscv_vzext_vf4_u32m8(va, vlmax));
    vint32m8_t vb32 = __riscv_vreinterpret_v_u32m8_i32m8(__riscv_vzext_vf4_u32m8(vb, vlmax));

    // Group MulPipe instructions (vmadd/vmacc).
    // Using LMUL=8 ensures long chime lengths, which helps hide the 3-stage 
    // Integer Multiply pipeline latency on the Saturn microarchitecture.
    vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_mult, v_bias, vlmax);
    vacc = __riscv_vmacc_vx_i32m8(vacc, b_mult, vb32, vlmax);

    // Narrowing operations (e32 -> e16 -> e8).
    // These instructions (vnclip, vnclipu) utilize the ShiftPipe. vnclip saturates to int16 (vnsra would wrap);
    // the output zero point is a saturating add, and output_min is applied in int16, which also clamps negatives.
    vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vlmax);
    vacc16 = __riscv_vsadd_vx_i16m4(vacc16, o_zp, vlmax);
    vacc16 = __riscv_vmax_vx_i16m4(vacc16, output_min, vlmax);
    vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RNU, vlmax);

    // Final clamp and store.
    // vminu utilizes the IntPipe.
    vout = __riscv_vminu_vx_u8m2(vout, output_max, vlmax);
    
    // Vector Store (VSS Sequencer)
    __riscv_vse8_v_u8m2(output + i, vout, vlmax);
  }

  // --- VLA Tail Handling ---
  // Processes remaining `batch % vlmax` elements using standard VLA (Vector Length Agnostic) patterns.
  if (i < batch) {
    size_t vl = __riscv_vsetvl_e8m2(batch - i);
    
    vuint8m2_t va = __riscv_vle8_v_u8m2(input_a + i, vl);
    vuint8m2_t vb = __riscv_vle8_v_u8m2(input_b + i, vl);

    // Maintain the same ILP strategy for the tail code.
    vint32m8_t va32 = __riscv_vreinterpret_v_u32m8_i32m8(__riscv_vzext_vf4_u32m8(va, vl));
    vint32m8_t vb32 = __riscv_vreinterpret_v_u32m8_i32m8(__riscv_vzext_vf4_u32m8(vb, vl));

    vint32m8_t vacc = __riscv_vmadd_vx_i32m8(va32, a_mult, v_bias, vl);
    vacc = __riscv_vmacc_vx_i32m8(vacc, b_mult, vb32, vl);

    vint16m4_t vacc16 = __riscv_vnclip_wx_i16m4(vacc, shift, __RISCV_VXRM_RDN, vl);
    vacc16 = __riscv_vsadd_vx_i16m4(vacc16, o_zp, vl);
    vacc16 = __riscv_vmax_vx_i16m4(vacc16, output_min, vl);
    vuint8m2_t vout = __riscv_vnclipu_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(vacc16), 0, __RISCV_VXRM_RNU, vl);
    
    vout = __riscv_vminu_vx_u8m2(vout, output_max, vl);
    
    __riscv_vse8_v_u8m2(output + i, vout, vl);
  }
}

