#pragma once
// Symbolic-scalar forms of intrinsics gen_sym_vx.py cannot derive (they have no .vv counterpart): the policy
// variants and the float merge. Each is the same composition as its concrete model (policy.hpp, core.hpp) with
// the scalar x splat into every lane, x converted to the intrinsic's scalar type as C converts an argument.
// Included after sym_vx_gen.hpp, so the unmasked symbolic forms they build on are declared.
#include "../intrinsics.hpp"
namespace salt_cvc5 {
// _mu: masked-off lanes < vl keep maskedoff (policy.hpp)
template<typename U> inline RVVVector __riscv_vfmin_vf_f32m2_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a, SymbolicScalar<U> b, size_t vl) { return rvv_vmerge_vvm_impl(mo, __riscv_vfmin_vf_f32m2(a, b, vl), m, vl, 32); }
template<typename U> inline RVVVector __riscv_vfmax_vf_f32m2_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a, SymbolicScalar<U> b, size_t vl) { return rvv_vmerge_vvm_impl(mo, __riscv_vfmax_vf_f32m2(a, b, vl), m, vl, 32); }
// vfmerge.vfm: lane i is x where the mask is set, a elsewhere (rvv_vfmerge_vfm_f32_impl)
template<typename U> inline vfloat32m1_t __riscv_vfmerge_vfm_f32m1(const vfloat32m1_t& a, SymbolicScalar<U> b, const MaskVector& mask, size_t vl) { return rvv_vmerge_vvm_impl(a, rvv_sym_splat_as<float>(b, vl), mask, vl, 32); }
template<typename U> inline vfloat32m2_t __riscv_vfmerge_vfm_f32m2(const vfloat32m2_t& a, SymbolicScalar<U> b, const MaskVector& mask, size_t vl) { return rvv_vmerge_vvm_impl(a, rvv_sym_splat_as<float>(b, vl), mask, vl, 32); }
// Masked vsmul.vx (_mu) with a symbolic scalar: the unmasked symbolic op, inactive lanes keep maskedoff.
template<typename U> inline RVVVector __riscv_vsmul_vx_i16m8_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i16m8(a, b, vxrm, vl), m, vl, 16);
}
template<typename U> inline RVVVector __riscv_vsmul_vx_i16m1_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i16m1(a, b, vxrm, vl), m, vl, 16);
}
template<typename U> inline RVVVector __riscv_vsmul_vx_i16m2_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i16m2(a, b, vxrm, vl), m, vl, 16);
}
template<typename U> inline RVVVector __riscv_vsmul_vx_i16m4_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i16m4(a, b, vxrm, vl), m, vl, 16);
}
template<typename U> inline RVVVector __riscv_vsmul_vx_i32m1_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i32m1(a, b, vxrm, vl), m, vl, 32);
}
template<typename U> inline RVVVector __riscv_vsmul_vx_i32m4_mu(const MaskVector& m, const RVVVector& mo, const RVVVector& a,
                                                               SymbolicScalar<U> b, unsigned vxrm, size_t vl) {
  return rvv_vmerge_vvm_impl(mo, __riscv_vsmul_vx_i32m4(a, b, vxrm, vl), m, vl, 32);
}
} // namespace salt_cvc5

namespace salt_cvc5 {
// vmv.s.x with a symbolic scalar: a reduction seeded with the output's prior value, read symbolically
// (`__riscv_vmv_s_x_u32m1(*output, vl)`, the read rewritten to salt_scalar). Element 0 is the scalar, as in the
// concrete model.
template<typename U> inline RVVVector __riscv_vmv_s_x_u32m1(SymbolicScalar<U> x, size_t vl) {
    RVVVector r = _gini_s_x(0, 32, vl); r.setElement(0, x.template cast<uint32_t>().term()); return r;
}
template<typename U> inline RVVVector __riscv_vmv_s_x_i32m1(SymbolicScalar<U> x, size_t vl) {
    RVVVector r = _gini_s_x(0, 32, vl); r.setElement(0, x.template cast<int32_t>().term()); return r;
}
} // namespace salt_cvc5
