#pragma once
// Per-element comparison dispatched by ElementKind — cvc5 port.

#include "context.hpp"
#include "symbolic_buffer.hpp"

#include <cvc5/cvc5.h>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace salt_cvc5 {

// FP-aware equality at sort (eb, sb): treats +0 == -0 (avoids spurious
// counterexamples from BV-rep ambiguity for FP zero), is bit-equivalent for
// normal values, and false for NaN-vs-anything except NaN-vs-NaN (which the
// `both_nan` clause accepts as the standard NaN-equivalent verification rule).
// SALT_FP_STRICT=1: bit-exact except NaN payloads, so +0 and -0 differ. (A NaN
// result's payload is unconstrained in the model; every other value has one
// encoding, so equal bits is equality of values.)
inline Term _fp_element_equal(TermManager& tm, Term a, Term b, uint32_t eb, uint32_t sb) {
    static const bool strict = [] { const char* s = std::getenv("SALT_FP_STRICT"); return s && *s == '1'; }();
    Op fp_op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {eb, sb});
    Term a_fp = tm.mkTerm(fp_op, {a});
    Term b_fp = tm.mkTerm(fp_op, {b});
    Term a_nan = tm.mkTerm(Kind::FLOATINGPOINT_IS_NAN, {a_fp});
    Term b_nan = tm.mkTerm(Kind::FLOATINGPOINT_IS_NAN, {b_fp});
    Term both_nan = tm.mkTerm(Kind::AND, {a_nan, b_nan});
    Term same = strict ? tm.mkTerm(Kind::EQUAL, {a, b}) : tm.mkTerm(Kind::FLOATINGPOINT_EQ, {a_fp, b_fp});
    return tm.mkTerm(Kind::OR, {both_nan, same});
}

// FP-aware tolerance: accepts a and b if their ULP distance is <= ulp_tol.
// Map each IEEE bit pattern to a monotonic signed key (adjacent floats differ by
// 1; +0/-0 collapse to the same key), then bound |key(a) - key(b)|. NaN is still
// exact (both-NaN accepted, exactly-one-NaN rejected); Inf needs no special case
// (Inf-vs-finite is a huge ULP gap that no small tol admits). Everything but the
// NaN guard is pure BV, so the bitwuzla backend stays in its fast domain.
inline Term _fp_within_ulp(TermManager& tm, Term a, Term b,
                           uint32_t eb, uint32_t sb, uint64_t ulp_tol) {
    uint32_t W = eb + sb;
    Term zeroW = tm.mkBitVector(W, 0);
    Term sign_mask = tm.mkBitVector(W, (uint64_t)1 << (W - 1));
    // sign-bit-set == signed-less-than-zero of the raw bits
    Term a_neg = tm.mkTerm(Kind::BITVECTOR_SLT, {a, zeroW});
    Term b_neg = tm.mkTerm(Kind::BITVECTOR_SLT, {b, zeroW});
    Term key_a = tm.mkTerm(Kind::ITE, {a_neg, tm.mkTerm(Kind::BITVECTOR_SUB, {sign_mask, a}), a});
    Term key_b = tm.mkTerm(Kind::ITE, {b_neg, tm.mkTerm(Kind::BITVECTOR_SUB, {sign_mask, b}), b});
    // widen by one bit so the signed difference cannot overflow
    Op sext1 = tm.mkOp(Kind::BITVECTOR_SIGN_EXTEND, {1});
    Term diff = tm.mkTerm(Kind::BITVECTOR_SUB, {tm.mkTerm(sext1, {key_a}), tm.mkTerm(sext1, {key_b})});
    Term zeroW1 = tm.mkBitVector(W + 1, 0);
    Term is_neg = tm.mkTerm(Kind::BITVECTOR_SLT, {diff, zeroW1});
    Term absd = tm.mkTerm(Kind::ITE, {is_neg, tm.mkTerm(Kind::BITVECTOR_NEG, {diff}), diff});
    Term ulp_ok = tm.mkTerm(Kind::BITVECTOR_ULE, {absd, tm.mkBitVector(W + 1, ulp_tol)});
    Op fp_op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {eb, sb});
    Term a_nan = tm.mkTerm(Kind::FLOATINGPOINT_IS_NAN, {tm.mkTerm(fp_op, {a})});
    Term b_nan = tm.mkTerm(Kind::FLOATINGPOINT_IS_NAN, {tm.mkTerm(fp_op, {b})});
    Term both_nan = tm.mkTerm(Kind::AND, {a_nan, b_nan});
    Term neither_nan = tm.mkTerm(Kind::NOT, {tm.mkTerm(Kind::OR, {a_nan, b_nan})});
    return tm.mkTerm(Kind::OR, {both_nan, tm.mkTerm(Kind::AND, {neither_nan, ulp_ok})});
}

// ulp_tol == 0 keeps the exact bit-equivalent check (default). ulp_tol > 0
// relaxes FP outputs to a ULP bound; integer/mask outputs stay exact regardless
// (indices and masks are never "approximately" right).
inline Term element_equal(TermManager& tm, Term a, Term b, ElementKind kind, uint64_t ulp_tol = 0) {
    switch (kind) {
    case ElementKind::SINT:
    case ElementKind::UINT:
    case ElementKind::MASK: return tm.mkTerm(Kind::EQUAL, {a, b});
    case ElementKind::F16:  return ulp_tol ? _fp_within_ulp(tm, a, b, 5, 11, ulp_tol)  : _fp_element_equal(tm, a, b, 5, 11);
    case ElementKind::BF16: return ulp_tol ? _fp_within_ulp(tm, a, b, 8, 8,  ulp_tol)  : _fp_element_equal(tm, a, b, 8, 8);   // 1+8 exp + 7 mant
    case ElementKind::F32:  return ulp_tol ? _fp_within_ulp(tm, a, b, 8, 24, ulp_tol)  : _fp_element_equal(tm, a, b, 8, 24);
    case ElementKind::F64:  return ulp_tol ? _fp_within_ulp(tm, a, b, 11, 53, ulp_tol) : _fp_element_equal(tm, a, b, 11, 53);
    }
    __builtin_unreachable();
}

// The candidate's element matches the reference's under the NaN policy.
// SALT_NAN_POLICY=ref follows XNNPACK's binary-op oracle
// (test/vbinary-microkernel-tester.cc: "if (std::isnan(y_ref[i])) { /* not all
// our kernels do this */ }"): where the reference output is NaN, any candidate
// output is accepted. The default ("both", the unary-op oracle,
// test/vunary-microkernel-tester.h) requires a NaN there too. Integer elements
// are always exact. `ref` must be the reference's element.
inline Term element_matches_ref(TermManager& tm, Term ref, Term cand, ElementKind kind, uint64_t ulp_tol = 0) {
    static const bool ref_nan_free = [] { const char* s = std::getenv("SALT_NAN_POLICY"); return s && std::string(s) == "ref"; }();
    Term eq = element_equal(tm, ref, cand, kind, ulp_tol);
    if (!ref_nan_free) return eq;
    uint32_t eb = 0, sb = 0;
    switch (kind) {
    case ElementKind::F16:  eb = 5;  sb = 11; break;
    case ElementKind::BF16: eb = 8;  sb = 8;  break;
    case ElementKind::F32:  eb = 8;  sb = 24; break;
    case ElementKind::F64:  eb = 11; sb = 53; break;
    default: return eq;
    }
    Term ref_fp = tm.mkTerm(tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {eb, sb}), {ref});
    return tm.mkTerm(Kind::OR, {tm.mkTerm(Kind::FLOATINGPOINT_IS_NAN, {ref_fp}), eq});
}

// ---------------------------------------------------------------------------
// Buffer-level comparison: compare num_elements elements of elem_bytes width.
// Returns a single Bool term: conjunction of all per-element equalities.
// Mirror of src/verification/core/comparison.hpp's buffers_equal.
// ---------------------------------------------------------------------------
inline Term buffers_equal(TermManager& tm,
                           const SymbolicBuffer& a,
                           const SymbolicBuffer& b,
                           size_t num_elements,
                           size_t elem_bytes,
                           ElementKind kind,
                           uint64_t ulp_tol = 0) {
    size_t elem_bits = elem_bytes * 8;
    std::vector<Term> equalities;
    equalities.reserve(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        size_t offset = i * elem_bytes;
        Term a_elem = a.loadScalar(offset, elem_bits);
        Term b_elem = b.loadScalar(offset, elem_bits);
        equalities.push_back(element_matches_ref(tm, a_elem, b_elem, kind, ulp_tol));
    }

    if (equalities.size() == 1) return equalities[0];

    // AND all equalities together.
    Term result = equalities[0];
    for (size_t i = 1; i < equalities.size(); i++) {
        result = tm.mkTerm(Kind::AND, {result, equalities[i]});
    }
    return result;
}

// True if the reference never wrote the element at byte `offset`: every byte is
// still the term from `ref_init`, the shared initial image of both outputs.
inline bool ref_element_untouched(const SymbolicBuffer& ref, const std::vector<Term>& ref_init,
                                  size_t offset, size_t elem_bytes) {
    for (size_t j = 0; j < elem_bytes; j++) {
        if (offset + j >= ref_init.size() || ref.getByte(offset + j) != ref_init[offset + j]) return false;
    }
    return true;
}

// buffers_equal over the elements the reference wrote. Elements it left
// untouched are holes in its output (packw padding slots): the candidate may
// leave them or write them, as callers never read them. `pad(i)` marks layout
// padding (a packed tile's columns past nc), don't-care even where the
// reference writes it: NEON packw zero-fills the bias pad only when bias is NULL.
inline Term buffers_equal_written(TermManager& tm,
                                  const SymbolicBuffer& ref,
                                  const SymbolicBuffer& cand,
                                  size_t num_elements,
                                  size_t elem_bytes,
                                  ElementKind kind,
                                  uint64_t ulp_tol,
                                  const std::vector<Term>& ref_init,
                                  const std::function<bool(size_t)>& pad = nullptr) {
    size_t elem_bits = elem_bytes * 8;
    Term result = tm.mkTrue();
    for (size_t i = 0; i < num_elements; i++) {
        size_t offset = i * elem_bytes;
        if (ref_element_untouched(ref, ref_init, offset, elem_bytes) || (pad && pad(i))) continue;
        Term eq = element_matches_ref(tm, ref.loadScalar(offset, elem_bits), cand.loadScalar(offset, elem_bits),
                                kind, ulp_tol);
        result = tm.mkTerm(Kind::AND, {result, eq});
    }
    return result;
}

// Strided 2D logical output region: compares rows×cols elements at affine
// offsets r*row_stride + c*col_stride (bytes), ignoring everything in between
// (e.g. row padding). Used for non-contiguous outputs (transpose, GEMM).
inline Term buffers_equal_strided(TermManager& tm,
                                  const SymbolicBuffer& a,
                                  const SymbolicBuffer& b,
                                  size_t rows, size_t cols,
                                  size_t row_stride, size_t col_stride,
                                  size_t elem_bytes,
                                  ElementKind kind,
                                  uint64_t ulp_tol = 0) {
    size_t elem_bits = elem_bytes * 8;
    std::vector<Term> equalities;
    equalities.reserve(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            size_t off = r * row_stride + c * col_stride;
            equalities.push_back(element_matches_ref(tm, a.loadScalar(off, elem_bits),
                                               b.loadScalar(off, elem_bits), kind, ulp_tol));
        }
    }
    if (equalities.empty()) return tm.mkTrue();
    Term result = equalities[0];
    for (size_t i = 1; i < equalities.size(); i++) {
        result = tm.mkTerm(Kind::AND, {result, equalities[i]});
    }
    return result;
}

} // namespace salt_cvc5
