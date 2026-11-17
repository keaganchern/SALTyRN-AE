#pragma once
// Symbolic-scalar .vx forms (a symbolic param as the scalar operand). The spec
// defines op.vx(v, x) as op.vv(v, splat(x)), shift amounts masked alike, so each
// forwards to the .vv model with x in every lane: exact by definition.
#include "../intrinsics.hpp"
#include "int_arith.hpp"
#include "widening.hpp"
namespace salt_cvc5 {
template<typename T>
inline RVVVector rvv_sym_splat(SymbolicScalar<T> x, size_t vl) {
    return RVVVector(g_ctx->tm, SymTraits<T>::bits, std::vector<Term>(vl, x.term()));
}
inline RVVVector __riscv_vmaxu_vx_u8m8(const RVVVector& a, SymbolicScalar<uint8_t> b, size_t vl) { return __riscv_vmaxu_vv_u8m8(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vminu_vx_u8m8(const RVVVector& a, SymbolicScalar<uint8_t> b, size_t vl) { return __riscv_vminu_vv_u8m8(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vmaxu_vx_u8m2(const RVVVector& a, SymbolicScalar<uint8_t> b, size_t vl) { return __riscv_vmaxu_vv_u8m2(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vminu_vx_u8m2(const RVVVector& a, SymbolicScalar<uint8_t> b, size_t vl) { return __riscv_vminu_vv_u8m2(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vmax_vx_i8m8(const RVVVector& a, SymbolicScalar<int8_t> b, size_t vl) { return __riscv_vmax_vv_i8m8(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vmin_vx_i8m8(const RVVVector& a, SymbolicScalar<int8_t> b, size_t vl) { return __riscv_vmin_vv_i8m8(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vmax_vx_i16m4(const RVVVector& a, SymbolicScalar<int16_t> b, size_t vl) { return __riscv_vmax_vv_i16m4(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vmin_vx_i16m4(const RVVVector& a, SymbolicScalar<int16_t> b, size_t vl) { return __riscv_vmin_vv_i16m4(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vsub_vx_i16m4(const RVVVector& a, SymbolicScalar<int16_t> b, size_t vl) { return __riscv_vsub_vv_i16m4(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vwsubu_vx_u16m4(const RVVVector& a, SymbolicScalar<uint8_t> b, size_t vl) { return __riscv_vwsubu_vv_u16m4(a, rvv_sym_splat(b, vl), vl); }
inline RVVVector __riscv_vsll_vx_i32m8(const RVVVector& a, SymbolicScalar<size_t> s, size_t vl) {
    return __riscv_vsll_vv_i32m8(a, rvv_sym_splat(s.template cast<uint32_t>(), vl), vl);
}
} // namespace salt_cvc5
