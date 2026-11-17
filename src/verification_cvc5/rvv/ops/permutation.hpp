#pragma once
// RVV permutation: vslideup/down, vslide1up/down, vrgather(ei16), vcompress, vmerge (cvc5).
// Pure data-movement on raw BV lanes; element width from the name's SEW token.
#include "../intrinsics.hpp"
namespace salt_cvc5 {


// VLMAX of a type: SEW p, LMUL num/den. The slides and gathers read source elements up to VLMAX, whatever vl is
// (RVV spec 16.3, 16.4): a source index at or past VLMAX reads 0; one in a short source's tail is unspecified
// (rvv_lane_or_tail, core.hpp).
inline size_t _gperm_vlmax(size_t p, size_t num, size_t den){ size_t n = g_ctx->vlen * num / (p * den); return n ? n : 1; }
// vslideup: res[i] = i<offset ? dest[i] : src[i-offset].
inline RVVVector _gperm_slideup(const RVVVector& dest, const RVVVector& src, size_t off, size_t p, size_t vl, size_t vlmax){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl); (void)vlmax;
  for(size_t i=0;i<vl;i++) r.push_back(i<off ? rvv_lane_or_tail(dest,i,p) : rvv_lane_or_tail(src,i-off,p));
  return RVVVector(tm,p,r);
}
// vslidedown: res[i] = (i+offset<VLMAX) ? src[i+offset] : 0.
inline RVVVector _gperm_slidedown(const RVVVector& src, size_t off, size_t p, size_t vl, size_t vlmax){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++) r.push_back((i+off<vlmax) ? rvv_lane_or_tail(src,i+off,p) : mk_bv_val(tm,p,0));
  return RVVVector(tm,p,r);
}
// vslide1up: res[0]=scalar; res[i]=src[i-1].
inline RVVVector _gperm_slide1up(const RVVVector& src, Term sbv, size_t p, size_t vl){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++) r.push_back(i==0 ? sbv : rvv_lane_or_tail(src,i-1,p));
  return RVVVector(tm,p,r);
}
// vslide1down: res[i]=src[i+1] (i<vl-1); res[vl-1]=scalar.
inline RVVVector _gperm_slide1down(const RVVVector& src, Term sbv, size_t p, size_t vl){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++) r.push_back((i+1<vl) ? rvv_lane_or_tail(src,i+1,p) : sbv);
  return RVVVector(tm,p,r);
}
// vrgather_vv/ei16: res[i] = idx[i]<VLMAX ? op1[idx[i]] : 0; ITE chain over j in [0,VLMAX).
// idx_bits = width of an index element (SEW for _vv, 16 for ei16).
inline RVVVector _gperm_gather_v(const RVVVector& op1, const RVVVector& idx, size_t idx_bits, size_t p, size_t vl, size_t vlmax){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  std::vector<Term> src; src.reserve(vlmax);
  for(size_t j=0;j<vlmax;j++) src.push_back(rvv_lane_or_tail(op1,j,p));
  for(size_t i=0;i<vl;i++){
    Term acc=mk_bv_val(tm,p,0);
    Term ix=idx.getElement(i);
    for(size_t j=vlmax;j-->0;){
      if (idx_bits < 64 && j >> idx_bits) continue;              // an index the element width cannot hold
      Term sel=fold_eq(tm,ix,mk_bv_val(tm,idx_bits,(int64_t)j));
      acc=fold_ite(tm,sel,src[j],acc);
    }
    r.push_back(acc);
  }
  return RVVVector(tm,p,r);
}
// vrgather_vx: same concrete index for all lanes; res[i] = index<VLMAX ? op1[index] : 0.
inline RVVVector _gperm_gather_x(const RVVVector& op1, size_t index, size_t p, size_t vl, size_t vlmax){
  auto& tm=g_ctx->tm; std::vector<Term> r; r.reserve(vl);
  Term v = (index<vlmax) ? rvv_lane_or_tail(op1,index,p) : mk_bv_val(tm,p,0);
  for(size_t i=0;i<vl;i++) r.push_back(v);
  return RVVVector(tm,p,r);
}
// vcompress: pack src[i] where mask[i] set into low lanes, in order; the lanes past the packed ones are tail
// elements (spec 16.5), unspecified: fresh values (rvv_tail_lane), not 0.
// Symbolic mask: res[k] = sum over i of [ (prefix_popcount(i)==k) && mask[i] ] ? src[i] : skip.
// Built as nested ITE: for each output slot k, scan i ascending and pick the (k+1)-th set lane.
inline RVVVector _gperm_compress(const RVVVector& src, const MaskVector& mask, size_t p, size_t vl){
  auto& tm=g_ctx->tm; Term one=mk_bv_val(tm,1,1);
  // running count BV (width enough for vl); count[i] = number of set mask bits in [0,i).
  size_t cw = 0; { size_t t=vl; while(t){cw++; t>>=1;} if(cw==0) cw=1; }
  std::vector<Term> bsel; bsel.reserve(vl);   // 1-bit: mask[i] selected
  std::vector<Term> pre;  pre.reserve(vl);     // cw-bit: prefix count before i
  Term run=mk_bv_val(tm,cw,0);
  for(size_t i=0;i<vl;i++){
    pre.push_back(run);
    Term b=fold_eq(tm,mask.getBit(i),one);
    bsel.push_back(mask.getBit(i));
    run=fold_ite(tm,b,fold_bvadd(tm,run,mk_bv_val(tm,cw,1)),run);
  }
  std::vector<Term> r; r.reserve(vl);
  for(size_t k=0;k<vl;k++){
    Term acc=rvv_tail_lane(p);
    Term kbv=mk_bv_val(tm,cw,(int64_t)k);
    for(size_t i=vl;i-->0;){
      // lane i lands in slot k iff mask[i] set AND prefix-count(i)==k (nested ITE = backend-neutral AND).
      Term hit=fold_eq(tm,bsel[i],one);
      Term atk=fold_eq(tm,pre[i],kbv);
      acc=fold_ite(tm,hit,fold_ite(tm,atk,rvv_lane_or_tail(src,i,p),acc),acc);
    }
    r.push_back(acc);
  }
  return RVVVector(tm,p,r);
}
// vmerge_vvm: res[i] = mask[i] ? op2[i] : op1[i].
inline RVVVector _gperm_merge_vv(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t p, size_t vl){
  auto& tm=g_ctx->tm; Term one=mk_bv_val(tm,1,1); std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++){
    Term sel=fold_eq(tm,mask.getBit(i),one);
    r.push_back(fold_ite(tm,sel,rvv_lane_or_tail(op2,i,p),rvv_lane_or_tail(op1,i,p)));
  }
  return RVVVector(tm,p,r);
}
// vmerge_vxm: res[i] = mask[i] ? scalar : op1[i].
inline RVVVector _gperm_merge_vx(const RVVVector& op1, Term sbv, const MaskVector& mask, size_t p, size_t vl){
  auto& tm=g_ctx->tm; Term one=mk_bv_val(tm,1,1); std::vector<Term> r; r.reserve(vl);
  for(size_t i=0;i<vl;i++){
    Term sel=fold_eq(tm,mask.getBit(i),one);
    r.push_back(fold_ite(tm,sel,sbv,rvv_lane_or_tail(op1,i,p)));
  }
  return RVVVector(tm,p,r);
}

inline RVVVector __riscv_vcompress_vm_f16m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f16m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f16m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f16m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f16mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f16mf4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_f32m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_f32m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_f32m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_f32m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_f32mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_f64m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_f64m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_f64m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_f64m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_i16m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i16m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i16m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i16m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i16mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i16mf4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_i32m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_i32m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_i32m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_i32m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_i32mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_i64m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_i64m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_i64m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_i64m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_i8m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8mf4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_i8mf8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u16m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u16m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u16m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u16m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u16mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u16mf4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,16,vl); }
inline RVVVector __riscv_vcompress_vm_u32m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_u32m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_u32m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_u32m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_u32mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,32,vl); }
inline RVVVector __riscv_vcompress_vm_u64m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_u64m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_u64m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_u64m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,64,vl); }
inline RVVVector __riscv_vcompress_vm_u8m1(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8m2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8m4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8m8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8mf2(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8mf4(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vcompress_vm_u8mf8(const RVVVector& src, const MaskVector& mask, size_t vl){ return _gperm_compress(src,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_f16m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_f16mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_f16mf4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_f32m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_f32m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_f32mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_f64m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_f64m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_f64m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_f64m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_i16m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i16m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i16m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i16m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i16mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i16mf4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_i32m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_i32m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_i32m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_i32m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_i32mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_i64m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_i64m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_i64m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_i64m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_i8m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8mf4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_i8mf8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u16m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_u16m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_u16m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_u16mf4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,16,vl); }
inline RVVVector __riscv_vmerge_vvm_u32m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_u32m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_u32m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_u32mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,32,vl); }
inline RVVVector __riscv_vmerge_vvm_u64m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_u64m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_u64m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_u64m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,64,vl); }
inline RVVVector __riscv_vmerge_vvm_u8m1(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8m2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8m4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8m8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8mf2(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8mf4(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vvm_u8mf8(const RVVVector& op1, const RVVVector& op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vv(op1,op2,mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i16m1(const RVVVector& op1, int16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_i16m2(const RVVVector& op1, int16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_i16mf2(const RVVVector& op1, int16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_i16mf4(const RVVVector& op1, int16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_i32m1(const RVVVector& op1, int32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_i32m2(const RVVVector& op1, int32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_i32m4(const RVVVector& op1, int32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_i32m8(const RVVVector& op1, int32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_i32mf2(const RVVVector& op1, int32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_i64m1(const RVVVector& op1, int64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_i64m2(const RVVVector& op1, int64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_i64m4(const RVVVector& op1, int64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_i64m8(const RVVVector& op1, int64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_i8m1(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8m2(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8m4(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8m8(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8mf2(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8mf4(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_i8mf8(const RVVVector& op1, int8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u16m2(const RVVVector& op1, uint16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_u16mf4(const RVVVector& op1, uint16_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,16,(int64_t)op2),mask,16,vl); }
inline RVVVector __riscv_vmerge_vxm_u32m2(const RVVVector& op1, uint32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_u32m4(const RVVVector& op1, uint32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_u32mf2(const RVVVector& op1, uint32_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,32,(int64_t)op2),mask,32,vl); }
inline RVVVector __riscv_vmerge_vxm_u64m1(const RVVVector& op1, uint64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_u64m2(const RVVVector& op1, uint64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_u64m4(const RVVVector& op1, uint64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_u64m8(const RVVVector& op1, uint64_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,64,(int64_t)op2),mask,64,vl); }
inline RVVVector __riscv_vmerge_vxm_u8m1(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8m2(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8m4(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8m8(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8mf2(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8mf4(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vmerge_vxm_u8mf8(const RVVVector& op1, uint8_t op2, const MaskVector& mask, size_t vl){ return _gperm_merge_vx(op1,mk_bv_val(g_ctx->tm,8,(int64_t)op2),mask,8,vl); }
inline RVVVector __riscv_vrgather_vv_f16m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgather_vv_f16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vv_f16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vv_f16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vv_f16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vv_f16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vv_f32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vv_f32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vv_f32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vv_f64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vv_f64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vv_f64m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vv_f64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vv_i16m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgather_vv_i16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vv_i16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vv_i16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vv_i16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vv_i16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vv_i32m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgather_vv_i32m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgather_vv_i32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vv_i32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vv_i32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vv_i64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vv_i64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vv_i64m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vv_i64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vv_i8m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgather_vv_i8m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgather_vv_i8m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgather_vv_i8m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vrgather_vv_i8mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgather_vv_i8mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgather_vv_i8mf8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vrgather_vv_u16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vv_u16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vv_u16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vv_u16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vv_u16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vv_u32m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgather_vv_u32m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgather_vv_u32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vv_u32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vv_u32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,32,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vv_u64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vv_u64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vv_u64m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vv_u64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,64,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vv_u8m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgather_vv_u8m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgather_vv_u8m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgather_vv_u8mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgather_vv_u8mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgather_vv_u8mf8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,8,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vrgather_vx_f16m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgather_vx_f16m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vx_f16m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vx_f16m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vx_f16mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vx_f16mf4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vx_f32m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgather_vx_f32m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgather_vx_f32m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vx_f32m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vx_f32mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vx_f64m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vx_f64m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vx_f64m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vx_f64m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vx_i16m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgather_vx_i16m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vx_i16m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vx_i16m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vx_i16mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vx_i16mf4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vx_i32m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgather_vx_i32m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgather_vx_i32m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vx_i32m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vx_i32mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vx_i64m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vx_i64m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vx_i64m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vx_i64m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vx_i8m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgather_vx_i8m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgather_vx_i8m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgather_vx_i8m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vrgather_vx_i8mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgather_vx_i8mf4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgather_vx_i8mf8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vrgather_vx_u16m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgather_vx_u16m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgather_vx_u16m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgather_vx_u16m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgather_vx_u16mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgather_vx_u16mf4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgather_vx_u32m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgather_vx_u32m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgather_vx_u32m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgather_vx_u32m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgather_vx_u32mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgather_vx_u64m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgather_vx_u64m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgather_vx_u64m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgather_vx_u64m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgather_vx_u8m1(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgather_vx_u8m2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgather_vx_u8m4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgather_vx_u8m8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vrgather_vx_u8mf2(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgather_vx_u8mf4(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgather_vx_u8mf8(const RVVVector& op1, size_t index, size_t vl){ return _gperm_gather_x(op1,index,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vrgatherei16_vv_f16m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_f16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgatherei16_vv_f32m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f32m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_f64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_f64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i16m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_i16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgatherei16_vv_i32m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i32m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_i64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i64m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i8m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i8m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i8m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_i8mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_i8mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgatherei16_vv_i8mf8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vrgatherei16_vv_u16m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u16m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u16m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u16m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u16mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_u16mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vrgatherei16_vv_u32m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u32m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u32m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u32m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u32mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_u64m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u64m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u64m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u64m8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u8m1(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u8m2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u8m4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vrgatherei16_vv_u8mf2(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vrgatherei16_vv_u8mf4(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vrgatherei16_vv_u8mf8(const RVVVector& op1, const RVVVector& index, size_t vl){ return _gperm_gather_v(op1,index,16,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vslide1down_vx_i16m2(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_i16m4(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_i16m8(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_i16mf2(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_i16mf4(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_i32m2(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_i32m4(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_i32m8(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_i32mf2(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_i64m1(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_i64m2(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_i64m4(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_i64m8(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_i8m1(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8m2(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8m4(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8m8(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8mf2(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8mf4(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_i8mf8(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u16m2(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_u16m4(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_u16m8(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_u16mf2(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_u16mf4(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1down_vx_u32m2(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_u32m4(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_u32m8(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_u32mf2(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1down_vx_u64m1(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_u64m2(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_u64m4(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_u64m8(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1down_vx_u8m1(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8m2(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8m4(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8m8(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8mf2(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8mf4(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1down_vx_u8mf8(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1down(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i16m1(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i16m2(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i16m4(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i16m8(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i16mf2(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i16mf4(const RVVVector& src, int16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_i32m1(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_i32m2(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_i32m4(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_i32m8(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_i32mf2(const RVVVector& src, int32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_i64m1(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_i64m2(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_i64m4(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_i64m8(const RVVVector& src, int64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_i8m1(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8m2(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8m4(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8m8(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8mf2(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8mf4(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_i8mf8(const RVVVector& src, int8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u16m1(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u16m2(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u16m4(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u16m8(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u16mf2(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u16mf4(const RVVVector& src, uint16_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,16,(int64_t)value),16,vl); }
inline RVVVector __riscv_vslide1up_vx_u32m1(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_u32m2(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_u32m4(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_u32m8(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_u32mf2(const RVVVector& src, uint32_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,32,(int64_t)value),32,vl); }
inline RVVVector __riscv_vslide1up_vx_u64m1(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_u64m2(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_u64m4(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_u64m8(const RVVVector& src, uint64_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,64,(int64_t)value),64,vl); }
inline RVVVector __riscv_vslide1up_vx_u8m1(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8m2(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8m4(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8m8(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8mf2(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8mf4(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslide1up_vx_u8mf8(const RVVVector& src, uint8_t value, size_t vl){ return _gperm_slide1up(src,mk_bv_val(g_ctx->tm,8,(int64_t)value),8,vl); }
inline RVVVector __riscv_vslidedown_vx_f16m1(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vslidedown_vx_f16m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslidedown_vx_f16m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslidedown_vx_f16m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslidedown_vx_f16mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslidedown_vx_f16mf4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslidedown_vx_f32m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslidedown_vx_f32mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslidedown_vx_f64m1(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslidedown_vx_f64m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslidedown_vx_f64m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslidedown_vx_i16m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslidedown_vx_i16m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslidedown_vx_i16m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslidedown_vx_i16mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslidedown_vx_i16mf4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslidedown_vx_i32m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vslidedown_vx_i32m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslidedown_vx_i32mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslidedown_vx_i64m1(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslidedown_vx_i64m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vslidedown_vx_i64m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslidedown_vx_i64m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslidedown_vx_i8m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vslidedown_vx_i8m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vslidedown_vx_i8m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vslidedown_vx_i8mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vslidedown_vx_i8mf4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vslidedown_vx_i8mf8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vslidedown_vx_u16m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslidedown_vx_u16m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslidedown_vx_u16m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslidedown_vx_u16mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslidedown_vx_u16mf4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslidedown_vx_u32m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vslidedown_vx_u32m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslidedown_vx_u32mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslidedown_vx_u64m1(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslidedown_vx_u64m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vslidedown_vx_u64m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslidedown_vx_u64m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslidedown_vx_u8m2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vslidedown_vx_u8m4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vslidedown_vx_u8m8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vslidedown_vx_u8mf2(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vslidedown_vx_u8mf4(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vslidedown_vx_u8mf8(const RVVVector& src, size_t offset, size_t vl){ return _gperm_slidedown(src,offset,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vslideup_vx_f16m1(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,1)); }
inline RVVVector __riscv_vslideup_vx_f16m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslideup_vx_f16m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslideup_vx_f16m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslideup_vx_f16mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslideup_vx_f16mf4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslideup_vx_f32m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vslideup_vx_f32m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vslideup_vx_f32m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslideup_vx_f32mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslideup_vx_f64m1(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslideup_vx_f64m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vslideup_vx_f64m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslideup_vx_f64m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslideup_vx_i16m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslideup_vx_i16m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslideup_vx_i16m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslideup_vx_i16mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslideup_vx_i16mf4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslideup_vx_i32m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vslideup_vx_i32m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vslideup_vx_i32m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslideup_vx_i32mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslideup_vx_i64m1(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslideup_vx_i64m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vslideup_vx_i64m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslideup_vx_i64m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslideup_vx_i8m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vslideup_vx_i8m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vslideup_vx_i8m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vslideup_vx_i8mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vslideup_vx_i8mf4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vslideup_vx_i8mf8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,8)); }
inline RVVVector __riscv_vslideup_vx_u16m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,2,1)); }
inline RVVVector __riscv_vslideup_vx_u16m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,4,1)); }
inline RVVVector __riscv_vslideup_vx_u16m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,8,1)); }
inline RVVVector __riscv_vslideup_vx_u16mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,2)); }
inline RVVVector __riscv_vslideup_vx_u16mf4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,16,vl,_gperm_vlmax(16,1,4)); }
inline RVVVector __riscv_vslideup_vx_u32m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,2,1)); }
inline RVVVector __riscv_vslideup_vx_u32m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,4,1)); }
inline RVVVector __riscv_vslideup_vx_u32m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,8,1)); }
inline RVVVector __riscv_vslideup_vx_u32mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,32,vl,_gperm_vlmax(32,1,2)); }
inline RVVVector __riscv_vslideup_vx_u64m1(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,1,1)); }
inline RVVVector __riscv_vslideup_vx_u64m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,2,1)); }
inline RVVVector __riscv_vslideup_vx_u64m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,4,1)); }
inline RVVVector __riscv_vslideup_vx_u64m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,64,vl,_gperm_vlmax(64,8,1)); }
inline RVVVector __riscv_vslideup_vx_u8m2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,2,1)); }
inline RVVVector __riscv_vslideup_vx_u8m4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,4,1)); }
inline RVVVector __riscv_vslideup_vx_u8m8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,8,1)); }
inline RVVVector __riscv_vslideup_vx_u8mf2(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,2)); }
inline RVVVector __riscv_vslideup_vx_u8mf4(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,4)); }
inline RVVVector __riscv_vslideup_vx_u8mf8(const RVVVector& dest, const RVVVector& src, size_t offset, size_t vl){ return _gperm_slideup(dest,src,offset,8,vl,_gperm_vlmax(8,1,8)); }

} // namespace salt_cvc5
