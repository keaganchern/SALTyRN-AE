#pragma once
// RVV initialize/move family — generated (ns salt_cvc5).
// vmv_v_v (copy), vmv_v_x (broadcast), vmv_s_x (lane0=scalar,tail 0),
// vmv_x_s (lane0 -> scalar, returns 0 per tree convention), vundefined (zeros).
#include "../intrinsics.hpp"
namespace salt_cvc5 {
inline RVVVector _gini_copy(const RVVVector& s,size_t sew,size_t vl){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++) r.push_back(s.getElement(i));
  return RVVVector(tm,sew,r);
}
inline RVVVector _gini_bcast(int64_t v,size_t sew,size_t vl){
  auto& tm=g_ctx->tm; Term b=mk_bv_val(tm,sew,v);
  return RVVVector(tm,sew,std::vector<Term>(vl,b));
}
inline RVVVector _gini_s_x(int64_t v,size_t sew,size_t vl){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  r.push_back(mk_bv_val(tm,sew,v));
  for(size_t i=1;i<vl;i++) r.push_back(rvv_tail_lane(sew));   // vmv.s.x writes element 0; the rest are tail
  return RVVVector(tm,sew,r);
}
// An undefined register (vundefined): every element a fresh unconstrained value, so a kernel whose outputs depend
// on one is not proved against a convenient 0 (hardware leaves whatever the register held).
inline RVVVector _gini_undef(size_t sew){
  auto& tm=g_ctx->tm; size_t n=g_ctx->vlen/sew*8; if(n<1) n=1;
  static size_t undef_count = 0;
  Sort s = tm.mkBitVectorSort(static_cast<uint32_t>(sew));
  std::vector<Term> r; r.reserve(n);
  for(size_t i=0;i<n;i++) r.push_back(tm.mkConst(s, "_rvv_undef_" + std::to_string(undef_count++)));
  return RVVVector(tm,sew,r);
}
inline RVVVector __riscv_vmv_s_x_i16m1(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i16m2(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i16m4(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i16m8(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i16mf2(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i16mf4(int16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_i32m1(int32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_i32m2(int32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_i32m4(int32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_i32m8(int32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_i32mf2(int32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_i64m1(int64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_i64m2(int64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_i64m4(int64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_i64m8(int64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_i8m1(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8m2(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8m4(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8m8(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8mf2(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8mf4(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_i8mf8(int8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u16m1(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u16m2(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u16m4(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u16m8(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u16mf2(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u16mf4(uint16_t src, size_t vl){ return _gini_s_x((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_s_x_u32m1(uint32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_u32m2(uint32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_u32m4(uint32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_u32m8(uint32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_u32mf2(uint32_t src, size_t vl){ return _gini_s_x((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_s_x_u64m1(uint64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_u64m2(uint64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_u64m4(uint64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_u64m8(uint64_t src, size_t vl){ return _gini_s_x((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_s_x_u8m1(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8m2(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8m4(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8m8(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8mf2(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8mf4(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_s_x_u8mf8(uint8_t src, size_t vl){ return _gini_s_x((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_v_f16m1(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_f16m2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_f16mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_f16mf4(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_f32m1(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_f32m2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_f32m8(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_f32mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_f64m1(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_f64m2(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_f64m4(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_f64m8(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_i16m1(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i16m2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i16m4(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i16m8(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i16mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i16mf4(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_i32m1(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_i32m2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_i32m4(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_i32m8(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_i32mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_i64m1(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_i64m2(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_i64m4(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_i64m8(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_i8m1(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8m2(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8m4(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8m8(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8mf4(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_i8mf8(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u16m1(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u16m2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u16m4(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u16m8(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u16mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u16mf4(const RVVVector& src, size_t vl){ return _gini_copy(src,16,vl); }
inline RVVVector __riscv_vmv_v_v_u32m1(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_u32m2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_u32m4(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_u32m8(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_u32mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,32,vl); }
inline RVVVector __riscv_vmv_v_v_u64m1(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_u64m2(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_u64m4(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_u64m8(const RVVVector& src, size_t vl){ return _gini_copy(src,64,vl); }
inline RVVVector __riscv_vmv_v_v_u8m1(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8m2(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8m4(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8m8(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8mf2(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8mf4(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_v_u8mf8(const RVVVector& src, size_t vl){ return _gini_copy(src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i16m1(int16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_i16m2(int16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_i16m8(int16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_i16mf2(int16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_i16mf4(int16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_i32m1(int32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_i32m2(int32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_i32m4(int32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_i32mf2(int32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_i64m1(int64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_i64m2(int64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_i64m4(int64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_i64m8(int64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_i8m1(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8m2(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8m4(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8m8(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8mf2(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8mf4(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_i8mf8(int8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u16m2(uint16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_u16mf2(uint16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_u16mf4(uint16_t src, size_t vl){ return _gini_bcast((int64_t)src,16,vl); }
inline RVVVector __riscv_vmv_v_x_u32m2(uint32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_u32m4(uint32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_u32mf2(uint32_t src, size_t vl){ return _gini_bcast((int64_t)src,32,vl); }
inline RVVVector __riscv_vmv_v_x_u64m1(uint64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_u64m2(uint64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_u64m4(uint64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_u64m8(uint64_t src, size_t vl){ return _gini_bcast((int64_t)src,64,vl); }
inline RVVVector __riscv_vmv_v_x_u8m1(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8m2(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8m4(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8m8(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8mf2(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8mf4(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline RVVVector __riscv_vmv_v_x_u8mf8(uint8_t src, size_t vl){ return _gini_bcast((int64_t)src,8,vl); }
inline int16_t __riscv_vmv_x_s_i16m2_i16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i16m2_i16"); }
inline int16_t __riscv_vmv_x_s_i16m4_i16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i16m4_i16"); }
inline int16_t __riscv_vmv_x_s_i16m8_i16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i16m8_i16"); }
inline int16_t __riscv_vmv_x_s_i16mf2_i16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i16mf2_i16"); }
inline int16_t __riscv_vmv_x_s_i16mf4_i16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i16mf4_i16"); }
inline int32_t __riscv_vmv_x_s_i32m1_i32(const RVVVector& src){ g_ctx->set_scalar_result(src.getElement(0)); return static_cast<int32_t>(kExtractMarker); }  // rsum: capture reduction sum
inline int32_t __riscv_vmv_x_s_i32m2_i32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i32m2_i32"); }
inline int32_t __riscv_vmv_x_s_i32m4_i32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i32m4_i32"); }
inline int32_t __riscv_vmv_x_s_i32m8_i32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i32m8_i32"); }
inline int32_t __riscv_vmv_x_s_i32mf2_i32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i32mf2_i32"); }
inline int64_t __riscv_vmv_x_s_i64m1_i64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i64m1_i64"); }
inline int64_t __riscv_vmv_x_s_i64m2_i64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i64m2_i64"); }
inline int64_t __riscv_vmv_x_s_i64m4_i64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i64m4_i64"); }
inline int64_t __riscv_vmv_x_s_i64m8_i64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i64m8_i64"); }
inline int8_t __riscv_vmv_x_s_i8m1_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8m1_i8"); }
inline int8_t __riscv_vmv_x_s_i8m2_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8m2_i8"); }
inline int8_t __riscv_vmv_x_s_i8m4_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8m4_i8"); }
inline int8_t __riscv_vmv_x_s_i8m8_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8m8_i8"); }
inline int8_t __riscv_vmv_x_s_i8mf2_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8mf2_i8"); }
inline int8_t __riscv_vmv_x_s_i8mf4_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8mf4_i8"); }
inline int8_t __riscv_vmv_x_s_i8mf8_i8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_i8mf8_i8"); }
inline uint16_t __riscv_vmv_x_s_u16m1_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16m1_u16"); }
inline uint16_t __riscv_vmv_x_s_u16m2_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16m2_u16"); }
inline uint16_t __riscv_vmv_x_s_u16m4_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16m4_u16"); }
inline uint16_t __riscv_vmv_x_s_u16m8_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16m8_u16"); }
inline uint16_t __riscv_vmv_x_s_u16mf2_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16mf2_u16"); }
inline uint16_t __riscv_vmv_x_s_u16mf4_u16(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u16mf4_u16"); }
inline uint32_t __riscv_vmv_x_s_u32m1_u32(const RVVVector& src){ g_ctx->set_scalar_result(src.getElement(0)); return static_cast<uint32_t>(kExtractMarker); }  // rsum: capture reduction sum
inline uint32_t __riscv_vmv_x_s_u32m2_u32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u32m2_u32"); }
inline uint32_t __riscv_vmv_x_s_u32m4_u32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u32m4_u32"); }
inline uint32_t __riscv_vmv_x_s_u32m8_u32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u32m8_u32"); }
inline uint32_t __riscv_vmv_x_s_u32mf2_u32(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u32mf2_u32"); }
inline uint64_t __riscv_vmv_x_s_u64m1_u64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u64m1_u64"); }
inline uint64_t __riscv_vmv_x_s_u64m2_u64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u64m2_u64"); }
inline uint64_t __riscv_vmv_x_s_u64m4_u64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u64m4_u64"); }
inline uint64_t __riscv_vmv_x_s_u64m8_u64(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u64m8_u64"); }
inline uint8_t __riscv_vmv_x_s_u8m1_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8m1_u8"); }
inline uint8_t __riscv_vmv_x_s_u8m2_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8m2_u8"); }
inline uint8_t __riscv_vmv_x_s_u8m4_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8m4_u8"); }
inline uint8_t __riscv_vmv_x_s_u8m8_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8m8_u8"); }
inline uint8_t __riscv_vmv_x_s_u8mf2_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8mf2_u8"); }
inline uint8_t __riscv_vmv_x_s_u8mf4_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8mf4_u8"); }
inline uint8_t __riscv_vmv_x_s_u8mf8_u8(const RVVVector& src){ salt_unmodeled("__riscv_vmv_x_s_u8mf8_u8"); }
inline RVVVector __riscv_vundefined_f16m1(){ return _gini_undef(16); }
inline vfloat16m1x2_t __riscv_vundefined_f16m1x2(){ vfloat16m1x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x3_t __riscv_vundefined_f16m1x3(){ vfloat16m1x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x4_t __riscv_vundefined_f16m1x4(){ vfloat16m1x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x5_t __riscv_vundefined_f16m1x5(){ vfloat16m1x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x6_t __riscv_vundefined_f16m1x6(){ vfloat16m1x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x7_t __riscv_vundefined_f16m1x7(){ vfloat16m1x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m1x8_t __riscv_vundefined_f16m1x8(){ vfloat16m1x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f16m2(){ return _gini_undef(16); }
inline vfloat16m2x2_t __riscv_vundefined_f16m2x2(){ vfloat16m2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m2x3_t __riscv_vundefined_f16m2x3(){ vfloat16m2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16m2x4_t __riscv_vundefined_f16m2x4(){ vfloat16m2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f16m4(){ return _gini_undef(16); }
inline vfloat16m4x2_t __riscv_vundefined_f16m4x2(){ vfloat16m4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f16m8(){ return _gini_undef(16); }
inline RVVVector __riscv_vundefined_f16mf2(){ return _gini_undef(16); }
inline vfloat16mf2x2_t __riscv_vundefined_f16mf2x2(){ vfloat16mf2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x3_t __riscv_vundefined_f16mf2x3(){ vfloat16mf2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x4_t __riscv_vundefined_f16mf2x4(){ vfloat16mf2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x5_t __riscv_vundefined_f16mf2x5(){ vfloat16mf2x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x6_t __riscv_vundefined_f16mf2x6(){ vfloat16mf2x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x7_t __riscv_vundefined_f16mf2x7(){ vfloat16mf2x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf2x8_t __riscv_vundefined_f16mf2x8(){ vfloat16mf2x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f16mf4(){ return _gini_undef(16); }
inline vfloat16mf4x2_t __riscv_vundefined_f16mf4x2(){ vfloat16mf4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x3_t __riscv_vundefined_f16mf4x3(){ vfloat16mf4x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x4_t __riscv_vundefined_f16mf4x4(){ vfloat16mf4x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x5_t __riscv_vundefined_f16mf4x5(){ vfloat16mf4x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x6_t __riscv_vundefined_f16mf4x6(){ vfloat16mf4x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x7_t __riscv_vundefined_f16mf4x7(){ vfloat16mf4x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vfloat16mf4x8_t __riscv_vundefined_f16mf4x8(){ vfloat16mf4x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f32m1(){ return _gini_undef(32); }
inline vfloat32m1x2_t __riscv_vundefined_f32m1x2(){ vfloat32m1x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m1x3_t __riscv_vundefined_f32m1x3(){ vfloat32m1x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m1x4_t __riscv_vundefined_f32m1x4(){ vfloat32m1x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m1x5_t __riscv_vundefined_f32m1x5(){ vfloat32m1x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
// Tuple types aliased to VectorTuple (intrinsics.hpp) return a tuple of zero vectors.
inline VectorTuple<RVVVector,6> __riscv_vundefined_f32m1x6(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,6>{{u,u,u,u,u,u}}; }
inline vfloat32m1x7_t __riscv_vundefined_f32m1x7(){ vfloat32m1x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m1x8_t __riscv_vundefined_f32m1x8(){ vfloat32m1x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f32m2(){ return _gini_undef(32); }
inline vfloat32m2x2_t __riscv_vundefined_f32m2x2(){ vfloat32m2x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m2x3_t __riscv_vundefined_f32m2x3(){ vfloat32m2x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32m2x4_t __riscv_vundefined_f32m2x4(){ vfloat32m2x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f32m4(){ return _gini_undef(32); }
inline vfloat32m4x2_t __riscv_vundefined_f32m4x2(){ vfloat32m4x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f32m8(){ return _gini_undef(32); }
inline RVVVector __riscv_vundefined_f32mf2(){ return _gini_undef(32); }
inline vfloat32mf2x2_t __riscv_vundefined_f32mf2x2(){ vfloat32mf2x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x3_t __riscv_vundefined_f32mf2x3(){ vfloat32mf2x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x4_t __riscv_vundefined_f32mf2x4(){ vfloat32mf2x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x5_t __riscv_vundefined_f32mf2x5(){ vfloat32mf2x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x6_t __riscv_vundefined_f32mf2x6(){ vfloat32mf2x6_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x7_t __riscv_vundefined_f32mf2x7(){ vfloat32mf2x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vfloat32mf2x8_t __riscv_vundefined_f32mf2x8(){ vfloat32mf2x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f64m1(){ return _gini_undef(64); }
inline vfloat64m1x2_t __riscv_vundefined_f64m1x2(){ vfloat64m1x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x3_t __riscv_vundefined_f64m1x3(){ vfloat64m1x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x4_t __riscv_vundefined_f64m1x4(){ vfloat64m1x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x5_t __riscv_vundefined_f64m1x5(){ vfloat64m1x5_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x6_t __riscv_vundefined_f64m1x6(){ vfloat64m1x6_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x7_t __riscv_vundefined_f64m1x7(){ vfloat64m1x7_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m1x8_t __riscv_vundefined_f64m1x8(){ vfloat64m1x8_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f64m2(){ return _gini_undef(64); }
inline vfloat64m2x2_t __riscv_vundefined_f64m2x2(){ vfloat64m2x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m2x3_t __riscv_vundefined_f64m2x3(){ vfloat64m2x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vfloat64m2x4_t __riscv_vundefined_f64m2x4(){ vfloat64m2x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f64m4(){ return _gini_undef(64); }
inline vfloat64m4x2_t __riscv_vundefined_f64m4x2(){ vfloat64m4x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_f64m8(){ return _gini_undef(64); }
inline RVVVector __riscv_vundefined_i16m1(){ return _gini_undef(16); }
inline vint16m1x2_t __riscv_vundefined_i16m1x2(){ vint16m1x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x3_t __riscv_vundefined_i16m1x3(){ vint16m1x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x4_t __riscv_vundefined_i16m1x4(){ vint16m1x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x5_t __riscv_vundefined_i16m1x5(){ vint16m1x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x6_t __riscv_vundefined_i16m1x6(){ vint16m1x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x7_t __riscv_vundefined_i16m1x7(){ vint16m1x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m1x8_t __riscv_vundefined_i16m1x8(){ vint16m1x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i16m2(){ return _gini_undef(16); }
inline vint16m2x2_t __riscv_vundefined_i16m2x2(){ vint16m2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m2x3_t __riscv_vundefined_i16m2x3(){ vint16m2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16m2x4_t __riscv_vundefined_i16m2x4(){ vint16m2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i16m4(){ return _gini_undef(16); }
inline vint16m4x2_t __riscv_vundefined_i16m4x2(){ vint16m4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i16m8(){ return _gini_undef(16); }
inline RVVVector __riscv_vundefined_i16mf2(){ return _gini_undef(16); }
inline vint16mf2x2_t __riscv_vundefined_i16mf2x2(){ vint16mf2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x3_t __riscv_vundefined_i16mf2x3(){ vint16mf2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x4_t __riscv_vundefined_i16mf2x4(){ vint16mf2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x5_t __riscv_vundefined_i16mf2x5(){ vint16mf2x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x6_t __riscv_vundefined_i16mf2x6(){ vint16mf2x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x7_t __riscv_vundefined_i16mf2x7(){ vint16mf2x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf2x8_t __riscv_vundefined_i16mf2x8(){ vint16mf2x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i16mf4(){ return _gini_undef(16); }
inline vint16mf4x2_t __riscv_vundefined_i16mf4x2(){ vint16mf4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x3_t __riscv_vundefined_i16mf4x3(){ vint16mf4x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x4_t __riscv_vundefined_i16mf4x4(){ vint16mf4x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x5_t __riscv_vundefined_i16mf4x5(){ vint16mf4x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x6_t __riscv_vundefined_i16mf4x6(){ vint16mf4x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x7_t __riscv_vundefined_i16mf4x7(){ vint16mf4x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vint16mf4x8_t __riscv_vundefined_i16mf4x8(){ vint16mf4x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i32m1(){ return _gini_undef(32); }
inline vint32m1x2_t __riscv_vundefined_i32m1x2(){ vint32m1x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x3_t __riscv_vundefined_i32m1x3(){ vint32m1x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x4_t __riscv_vundefined_i32m1x4(){ vint32m1x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x5_t __riscv_vundefined_i32m1x5(){ vint32m1x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x6_t __riscv_vundefined_i32m1x6(){ vint32m1x6_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x7_t __riscv_vundefined_i32m1x7(){ vint32m1x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m1x8_t __riscv_vundefined_i32m1x8(){ vint32m1x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i32m2(){ return _gini_undef(32); }
inline vint32m2x2_t __riscv_vundefined_i32m2x2(){ vint32m2x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m2x3_t __riscv_vundefined_i32m2x3(){ vint32m2x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32m2x4_t __riscv_vundefined_i32m2x4(){ vint32m2x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i32m4(){ return _gini_undef(32); }
inline vint32m4x2_t __riscv_vundefined_i32m4x2(){ vint32m4x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i32m8(){ return _gini_undef(32); }
inline RVVVector __riscv_vundefined_i32mf2(){ return _gini_undef(32); }
inline vint32mf2x2_t __riscv_vundefined_i32mf2x2(){ vint32mf2x2_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x3_t __riscv_vundefined_i32mf2x3(){ vint32mf2x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x4_t __riscv_vundefined_i32mf2x4(){ vint32mf2x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x5_t __riscv_vundefined_i32mf2x5(){ vint32mf2x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x6_t __riscv_vundefined_i32mf2x6(){ vint32mf2x6_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x7_t __riscv_vundefined_i32mf2x7(){ vint32mf2x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vint32mf2x8_t __riscv_vundefined_i32mf2x8(){ vint32mf2x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i64m1(){ return _gini_undef(64); }
inline vint64m1x2_t __riscv_vundefined_i64m1x2(){ vint64m1x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x3_t __riscv_vundefined_i64m1x3(){ vint64m1x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x4_t __riscv_vundefined_i64m1x4(){ vint64m1x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x5_t __riscv_vundefined_i64m1x5(){ vint64m1x5_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x6_t __riscv_vundefined_i64m1x6(){ vint64m1x6_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x7_t __riscv_vundefined_i64m1x7(){ vint64m1x7_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m1x8_t __riscv_vundefined_i64m1x8(){ vint64m1x8_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i64m2(){ return _gini_undef(64); }
inline vint64m2x2_t __riscv_vundefined_i64m2x2(){ vint64m2x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m2x3_t __riscv_vundefined_i64m2x3(){ vint64m2x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vint64m2x4_t __riscv_vundefined_i64m2x4(){ vint64m2x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i64m4(){ return _gini_undef(64); }
inline vint64m4x2_t __riscv_vundefined_i64m4x2(){ vint64m4x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i64m8(){ return _gini_undef(64); }
inline RVVVector __riscv_vundefined_i8m1(){ return _gini_undef(8); }
inline vint8m1x2_t __riscv_vundefined_i8m1x2(){ vint8m1x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x3_t __riscv_vundefined_i8m1x3(){ vint8m1x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x4_t __riscv_vundefined_i8m1x4(){ vint8m1x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x5_t __riscv_vundefined_i8m1x5(){ vint8m1x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x6_t __riscv_vundefined_i8m1x6(){ vint8m1x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x7_t __riscv_vundefined_i8m1x7(){ vint8m1x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m1x8_t __riscv_vundefined_i8m1x8(){ vint8m1x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i8m2(){ return _gini_undef(8); }
inline vint8m2x2_t __riscv_vundefined_i8m2x2(){ vint8m2x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m2x3_t __riscv_vundefined_i8m2x3(){ vint8m2x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8m2x4_t __riscv_vundefined_i8m2x4(){ vint8m2x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i8m4(){ return _gini_undef(8); }
inline vint8m4x2_t __riscv_vundefined_i8m4x2(){ vint8m4x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i8m8(){ return _gini_undef(8); }
inline RVVVector __riscv_vundefined_i8mf2(){ return _gini_undef(8); }
inline vint8mf2x2_t __riscv_vundefined_i8mf2x2(){ vint8mf2x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x3_t __riscv_vundefined_i8mf2x3(){ vint8mf2x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x4_t __riscv_vundefined_i8mf2x4(){ vint8mf2x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x5_t __riscv_vundefined_i8mf2x5(){ vint8mf2x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x6_t __riscv_vundefined_i8mf2x6(){ vint8mf2x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x7_t __riscv_vundefined_i8mf2x7(){ vint8mf2x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf2x8_t __riscv_vundefined_i8mf2x8(){ vint8mf2x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i8mf4(){ return _gini_undef(8); }
inline vint8mf4x2_t __riscv_vundefined_i8mf4x2(){ vint8mf4x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x3_t __riscv_vundefined_i8mf4x3(){ vint8mf4x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x4_t __riscv_vundefined_i8mf4x4(){ vint8mf4x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x5_t __riscv_vundefined_i8mf4x5(){ vint8mf4x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x6_t __riscv_vundefined_i8mf4x6(){ vint8mf4x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x7_t __riscv_vundefined_i8mf4x7(){ vint8mf4x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf4x8_t __riscv_vundefined_i8mf4x8(){ vint8mf4x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_i8mf8(){ return _gini_undef(8); }
inline vint8mf8x2_t __riscv_vundefined_i8mf8x2(){ vint8mf8x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x3_t __riscv_vundefined_i8mf8x3(){ vint8mf8x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x4_t __riscv_vundefined_i8mf8x4(){ vint8mf8x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x5_t __riscv_vundefined_i8mf8x5(){ vint8mf8x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x6_t __riscv_vundefined_i8mf8x6(){ vint8mf8x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x7_t __riscv_vundefined_i8mf8x7(){ vint8mf8x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vint8mf8x8_t __riscv_vundefined_i8mf8x8(){ vint8mf8x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u16m1(){ return _gini_undef(16); }
inline vuint16m1x2_t __riscv_vundefined_u16m1x2(){ vuint16m1x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x3_t __riscv_vundefined_u16m1x3(){ vuint16m1x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x4_t __riscv_vundefined_u16m1x4(){ vuint16m1x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x5_t __riscv_vundefined_u16m1x5(){ vuint16m1x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x6_t __riscv_vundefined_u16m1x6(){ vuint16m1x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x7_t __riscv_vundefined_u16m1x7(){ vuint16m1x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m1x8_t __riscv_vundefined_u16m1x8(){ vuint16m1x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u16m2(){ return _gini_undef(16); }
inline vuint16m2x2_t __riscv_vundefined_u16m2x2(){ vuint16m2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m2x3_t __riscv_vundefined_u16m2x3(){ vuint16m2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16m2x4_t __riscv_vundefined_u16m2x4(){ vuint16m2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u16m4(){ return _gini_undef(16); }
inline vuint16m4x2_t __riscv_vundefined_u16m4x2(){ vuint16m4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u16m8(){ return _gini_undef(16); }
inline RVVVector __riscv_vundefined_u16mf2(){ return _gini_undef(16); }
inline vuint16mf2x2_t __riscv_vundefined_u16mf2x2(){ vuint16mf2x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x3_t __riscv_vundefined_u16mf2x3(){ vuint16mf2x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x4_t __riscv_vundefined_u16mf2x4(){ vuint16mf2x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x5_t __riscv_vundefined_u16mf2x5(){ vuint16mf2x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x6_t __riscv_vundefined_u16mf2x6(){ vuint16mf2x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x7_t __riscv_vundefined_u16mf2x7(){ vuint16mf2x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf2x8_t __riscv_vundefined_u16mf2x8(){ vuint16mf2x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u16mf4(){ return _gini_undef(16); }
inline vuint16mf4x2_t __riscv_vundefined_u16mf4x2(){ vuint16mf4x2_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x3_t __riscv_vundefined_u16mf4x3(){ vuint16mf4x3_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x4_t __riscv_vundefined_u16mf4x4(){ vuint16mf4x4_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x5_t __riscv_vundefined_u16mf4x5(){ vuint16mf4x5_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x6_t __riscv_vundefined_u16mf4x6(){ vuint16mf4x6_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x7_t __riscv_vundefined_u16mf4x7(){ vuint16mf4x7_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline vuint16mf4x8_t __riscv_vundefined_u16mf4x8(){ vuint16mf4x8_t r; for (auto& e : r.val) e = _gini_undef(16); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u32m1(){ return _gini_undef(32); }
inline VectorTuple<RVVVector,2> __riscv_vundefined_u32m1x2(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,2>{{u,u}}; }
inline vuint32m1x3_t __riscv_vundefined_u32m1x3(){ vuint32m1x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32m1x4_t __riscv_vundefined_u32m1x4(){ vuint32m1x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32m1x5_t __riscv_vundefined_u32m1x5(){ vuint32m1x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32m1x6_t __riscv_vundefined_u32m1x6(){ vuint32m1x6_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32m1x7_t __riscv_vundefined_u32m1x7(){ vuint32m1x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32m1x8_t __riscv_vundefined_u32m1x8(){ vuint32m1x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u32m2(){ return _gini_undef(32); }
inline VectorTuple<RVVVector,2> __riscv_vundefined_u32m2x2(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,2>{{u,u}}; }
inline VectorTuple<RVVVector,3> __riscv_vundefined_u32m2x3(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,3>{{u,u,u}}; }
inline VectorTuple<RVVVector,4> __riscv_vundefined_u32m2x4(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,4>{{u,u,u,u}}; }
inline RVVVector __riscv_vundefined_u32m4(){ return _gini_undef(32); }
inline VectorTuple<RVVVector,2> __riscv_vundefined_u32m4x2(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,2>{{u,u}}; }
inline RVVVector __riscv_vundefined_u32m8(){ return _gini_undef(32); }
inline RVVVector __riscv_vundefined_u32mf2(){ return _gini_undef(32); }
inline VectorTuple<RVVVector,2> __riscv_vundefined_u32mf2x2(){ RVVVector u=_gini_undef(32); return VectorTuple<RVVVector,2>{{u,u}}; }
inline vuint32mf2x3_t __riscv_vundefined_u32mf2x3(){ vuint32mf2x3_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32mf2x4_t __riscv_vundefined_u32mf2x4(){ vuint32mf2x4_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32mf2x5_t __riscv_vundefined_u32mf2x5(){ vuint32mf2x5_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32mf2x6_t __riscv_vundefined_u32mf2x6(){ vuint32mf2x6_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32mf2x7_t __riscv_vundefined_u32mf2x7(){ vuint32mf2x7_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline vuint32mf2x8_t __riscv_vundefined_u32mf2x8(){ vuint32mf2x8_t r; for (auto& e : r.val) e = _gini_undef(32); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u64m1(){ return _gini_undef(64); }
inline vuint64m1x2_t __riscv_vundefined_u64m1x2(){ vuint64m1x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x3_t __riscv_vundefined_u64m1x3(){ vuint64m1x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x4_t __riscv_vundefined_u64m1x4(){ vuint64m1x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x5_t __riscv_vundefined_u64m1x5(){ vuint64m1x5_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x6_t __riscv_vundefined_u64m1x6(){ vuint64m1x6_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x7_t __riscv_vundefined_u64m1x7(){ vuint64m1x7_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m1x8_t __riscv_vundefined_u64m1x8(){ vuint64m1x8_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u64m2(){ return _gini_undef(64); }
inline vuint64m2x2_t __riscv_vundefined_u64m2x2(){ vuint64m2x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m2x3_t __riscv_vundefined_u64m2x3(){ vuint64m2x3_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline vuint64m2x4_t __riscv_vundefined_u64m2x4(){ vuint64m2x4_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u64m4(){ return _gini_undef(64); }
inline vuint64m4x2_t __riscv_vundefined_u64m4x2(){ vuint64m4x2_t r; for (auto& e : r.val) e = _gini_undef(64); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u64m8(){ return _gini_undef(64); }
inline RVVVector __riscv_vundefined_u8m1(){ return _gini_undef(8); }
inline vuint8m1x2_t __riscv_vundefined_u8m1x2(){ vuint8m1x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x3_t __riscv_vundefined_u8m1x3(){ vuint8m1x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x4_t __riscv_vundefined_u8m1x4(){ vuint8m1x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x5_t __riscv_vundefined_u8m1x5(){ vuint8m1x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x6_t __riscv_vundefined_u8m1x6(){ vuint8m1x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x7_t __riscv_vundefined_u8m1x7(){ vuint8m1x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m1x8_t __riscv_vundefined_u8m1x8(){ vuint8m1x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m2x2_t __riscv_vundefined_u8m2x2(){ vuint8m2x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m2x3_t __riscv_vundefined_u8m2x3(){ vuint8m2x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8m2x4_t __riscv_vundefined_u8m2x4(){ vuint8m2x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u8m4(){ return _gini_undef(8); }
inline vuint8m4x2_t __riscv_vundefined_u8m4x2(){ vuint8m4x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u8m8(){ return _gini_undef(8); }
inline RVVVector __riscv_vundefined_u8mf2(){ return _gini_undef(8); }
inline vuint8mf2x2_t __riscv_vundefined_u8mf2x2(){ vuint8mf2x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x3_t __riscv_vundefined_u8mf2x3(){ vuint8mf2x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x4_t __riscv_vundefined_u8mf2x4(){ vuint8mf2x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x5_t __riscv_vundefined_u8mf2x5(){ vuint8mf2x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x6_t __riscv_vundefined_u8mf2x6(){ vuint8mf2x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x7_t __riscv_vundefined_u8mf2x7(){ vuint8mf2x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf2x8_t __riscv_vundefined_u8mf2x8(){ vuint8mf2x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u8mf4(){ return _gini_undef(8); }
inline vuint8mf4x2_t __riscv_vundefined_u8mf4x2(){ vuint8mf4x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x3_t __riscv_vundefined_u8mf4x3(){ vuint8mf4x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x4_t __riscv_vundefined_u8mf4x4(){ vuint8mf4x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x5_t __riscv_vundefined_u8mf4x5(){ vuint8mf4x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x6_t __riscv_vundefined_u8mf4x6(){ vuint8mf4x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x7_t __riscv_vundefined_u8mf4x7(){ vuint8mf4x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf4x8_t __riscv_vundefined_u8mf4x8(){ vuint8mf4x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline RVVVector __riscv_vundefined_u8mf8(){ return _gini_undef(8); }
inline vuint8mf8x2_t __riscv_vundefined_u8mf8x2(){ vuint8mf8x2_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x3_t __riscv_vundefined_u8mf8x3(){ vuint8mf8x3_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x4_t __riscv_vundefined_u8mf8x4(){ vuint8mf8x4_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x5_t __riscv_vundefined_u8mf8x5(){ vuint8mf8x5_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x6_t __riscv_vundefined_u8mf8x6(){ vuint8mf8x6_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x7_t __riscv_vundefined_u8mf8x7(){ vuint8mf8x7_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
inline vuint8mf8x8_t __riscv_vundefined_u8mf8x8(){ vuint8mf8x8_t r; for (auto& e : r.val) e = _gini_undef(8); return r; }  // a tuple of undefined registers
} // namespace salt_cvc5
