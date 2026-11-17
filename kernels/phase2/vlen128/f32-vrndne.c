#include <riscv_vector.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

struct xnn_f32_default_params;

void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  const float magic_number = 8388608.0f;
  // Integer mask for clearing the sign bit
  const int32_t abs_mask = INT32_MAX; // 0x7FFFFFFF
  // Bit pattern of 2^23: for non-negative floats (and NaN) the integer order of the bits is the float order,
  // so |x| < 2^23 is a 1-stage integer compare. Values with |x| >= 2^23 (and Inf/NaN) are already integral.
  const int32_t magic_bits = 0x4B000000;

  while (n > 0) {
    // Set vector length for 32-bit elements and LMUL=8
    size_t vl = __riscv_vsetvl_e32m8(n);

    // Load input elements
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    // Absolute value on the 1-stage IntPipe: clear the sign bit
    vint32m8_t vabsx_int = __riscv_vand_vx_i32m8(__riscv_vreinterpret_v_f32m8_i32m8(vx), abs_mask, vl);
    vfloat32m8_t vabsx = __riscv_vreinterpret_v_i32m8_f32m8(vabsx_int);
    vbool4_t vsmall = __riscv_vmslt_vx_i32m8_b4(vabsx_int, magic_bits, vl);

    // The rounding trick: (|x| + magic_number) - magic_number, kept only where |x| < 2^23;
    // the other lanes pass |x| through (mask-undisturbed subtract)
    vfloat32m8_t vrndabsx = __riscv_vfadd_vf_f32m8(vabsx, magic_number, vl);
    vrndabsx = __riscv_vfsub_vf_f32m8_mu(vsmall, vabsx, vrndabsx, magic_number, vl);

    // Restore the original sign (one sign-injection op)
    vfloat32m8_t vy = __riscv_vfsgnj_vv_f32m8(vrndabsx, vx, vl);

    // Store the final rounded result
    __riscv_vse32_v_f32m8(output, vy, vl);

    // Update pointers and remaining count
    input += vl;
    output += vl;
    n -= vl;
  }
}
