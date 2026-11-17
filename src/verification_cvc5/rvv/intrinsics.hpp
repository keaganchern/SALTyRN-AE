#pragma once
// RVV shared infra base (cvc5): type aliases + vsetvl helper used by every
// intrinsic family. Every intrinsic — including the hand-written core — lives in
// ops/*.hpp and includes this. Intrinsic bodies follow op-semantics/rvv-intrinsics.md.

#include "../core/bv_fold.hpp"
#include "../core/context.hpp"
#include "../core/fp_convert.hpp"
#include "../core/symbolic_buffer.hpp"
#include "../core/symbolic_scalar.hpp"
#include "../core/symbolic_vector.hpp"

#include <cmath>
#include <cstring>
#include <cvc5/cvc5.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace salt_cvc5 {

// ===========================================================================
// Type aliases NOT defined in symbolic_vector.hpp (cvc5 tree currently only
// aliases the f16-conv-hwc2chw subset).  All these are RVVVector under the
// hood — adding `using X = RVVVector` here is harmless (and would also be
// harmless if `X` were already aliased elsewhere — same target).
// ===========================================================================
using vint8mf2_t   = RVVVector;
using vint8m1_t    = RVVVector;
using vint8m2_t    = RVVVector;
using vint8m4_t    = RVVVector;
using vint8m8_t    = RVVVector;
using vint16m1_t   = RVVVector;
using vint16m2_t   = RVVVector;
using vint16m4_t   = RVVVector;
using vint16m8_t   = RVVVector;
using vint16mf2_t  = RVVVector;
using vint32m1_t   = RVVVector;
using vint32m4_t   = RVVVector;
using vint32m8_t   = RVVVector;
using vuint8m1_t   = RVVVector;
using vuint8m2_t   = RVVVector;
using vuint8m4_t   = RVVVector;
using vuint8m8_t   = RVVVector;
using vuint8mf2_t  = RVVVector;
using vuint16m2_t  = RVVVector;
// Fractional LMUL (mf8/mf4/mf2): the AE VLEN-512 translation of qd8-f32-qc4w-gemm-minmax loads its 16 weight
// bytes as vint8mf4_t. The intrinsics are modeled (mem_unit, int_arith, widening); these are their types.
using vint8mf8_t    = RVVVector;
using vint8mf4_t    = RVVVector;
using vuint8mf8_t   = RVVVector;
using vuint8mf4_t   = RVVVector;
using vint16mf4_t   = RVVVector;
using vuint16mf4_t  = RVVVector;
using vint32mf2_t   = RVVVector;
using vuint32mf2_t  = RVVVector;
using vfloat16mf4_t = RVVVector;
using vfloat16mf2_t = RVVVector;
using vfloat32mf2_t = RVVVector;
using vuint16m4_t  = RVVVector;
using vuint32m1_t  = RVVVector;
using vuint32m4_t  = RVVVector;
using vuint32m8_t  = RVVVector;
using vfloat16m2_t = RVVVector;
using vfloat16m4_t = RVVVector;
using vfloat16m8_t = RVVVector;
using vfloat32m1_t = RVVVector;
using vfloat32m2_t = RVVVector;
using vfloat32m4_t = RVVVector;
using vfloat32m8_t = RVVVector;
using vuint32m1x2_t = VectorTuple<RVVVector, 2>;
using vuint32mf2x2_t = VectorTuple<RVVVector, 2>;
using vuint32m2x2_t = VectorTuple<RVVVector, 2>;
using vuint32m4x2_t = VectorTuple<RVVVector, 2>;
using vuint32m2x3_t = VectorTuple<RVVVector, 3>;
using vuint32m2x4_t = VectorTuple<RVVVector, 4>;
using vfloat32m1x6_t = VectorTuple<RVVVector, 6>;
using vfloat32m1x2_t = VectorTuple<RVVVector, 2>;
using vfloat32m1x3_t = VectorTuple<RVVVector, 3>;
using vfloat32m1x4_t = VectorTuple<RVVVector, 4>;
using vfloat32m1x5_t = VectorTuple<RVVVector, 5>;
using vfloat32m1x7_t = VectorTuple<RVVVector, 7>;
using vfloat32m1x8_t = VectorTuple<RVVVector, 8>;
// Every other segment tuple type (NF * LMUL <= 8): the same untyped VectorTuple.
using vint8mf8x2_t = VectorTuple<RVVVector, 2>;
using vint8mf8x3_t = VectorTuple<RVVVector, 3>;
using vint8mf8x4_t = VectorTuple<RVVVector, 4>;
using vint8mf8x5_t = VectorTuple<RVVVector, 5>;
using vint8mf8x6_t = VectorTuple<RVVVector, 6>;
using vint8mf8x7_t = VectorTuple<RVVVector, 7>;
using vint8mf8x8_t = VectorTuple<RVVVector, 8>;
using vint8mf4x2_t = VectorTuple<RVVVector, 2>;
using vint8mf4x3_t = VectorTuple<RVVVector, 3>;
using vint8mf4x4_t = VectorTuple<RVVVector, 4>;
using vint8mf4x5_t = VectorTuple<RVVVector, 5>;
using vint8mf4x6_t = VectorTuple<RVVVector, 6>;
using vint8mf4x7_t = VectorTuple<RVVVector, 7>;
using vint8mf4x8_t = VectorTuple<RVVVector, 8>;
using vint8mf2x2_t = VectorTuple<RVVVector, 2>;
using vint8mf2x3_t = VectorTuple<RVVVector, 3>;
using vint8mf2x4_t = VectorTuple<RVVVector, 4>;
using vint8mf2x5_t = VectorTuple<RVVVector, 5>;
using vint8mf2x6_t = VectorTuple<RVVVector, 6>;
using vint8mf2x7_t = VectorTuple<RVVVector, 7>;
using vint8mf2x8_t = VectorTuple<RVVVector, 8>;
using vint8m1x2_t = VectorTuple<RVVVector, 2>;
using vint8m1x3_t = VectorTuple<RVVVector, 3>;
using vint8m1x4_t = VectorTuple<RVVVector, 4>;
using vint8m1x5_t = VectorTuple<RVVVector, 5>;
using vint8m1x6_t = VectorTuple<RVVVector, 6>;
using vint8m1x7_t = VectorTuple<RVVVector, 7>;
using vint8m1x8_t = VectorTuple<RVVVector, 8>;
using vint8m2x2_t = VectorTuple<RVVVector, 2>;
using vint8m2x3_t = VectorTuple<RVVVector, 3>;
using vint8m2x4_t = VectorTuple<RVVVector, 4>;
using vint8m4x2_t = VectorTuple<RVVVector, 2>;
using vint16mf4x2_t = VectorTuple<RVVVector, 2>;
using vint16mf4x3_t = VectorTuple<RVVVector, 3>;
using vint16mf4x4_t = VectorTuple<RVVVector, 4>;
using vint16mf4x5_t = VectorTuple<RVVVector, 5>;
using vint16mf4x6_t = VectorTuple<RVVVector, 6>;
using vint16mf4x7_t = VectorTuple<RVVVector, 7>;
using vint16mf4x8_t = VectorTuple<RVVVector, 8>;
using vint16mf2x2_t = VectorTuple<RVVVector, 2>;
using vint16mf2x3_t = VectorTuple<RVVVector, 3>;
using vint16mf2x4_t = VectorTuple<RVVVector, 4>;
using vint16mf2x5_t = VectorTuple<RVVVector, 5>;
using vint16mf2x6_t = VectorTuple<RVVVector, 6>;
using vint16mf2x7_t = VectorTuple<RVVVector, 7>;
using vint16mf2x8_t = VectorTuple<RVVVector, 8>;
using vint16m1x2_t = VectorTuple<RVVVector, 2>;
using vint16m1x3_t = VectorTuple<RVVVector, 3>;
using vint16m1x4_t = VectorTuple<RVVVector, 4>;
using vint16m1x5_t = VectorTuple<RVVVector, 5>;
using vint16m1x6_t = VectorTuple<RVVVector, 6>;
using vint16m1x7_t = VectorTuple<RVVVector, 7>;
using vint16m1x8_t = VectorTuple<RVVVector, 8>;
using vint16m2x2_t = VectorTuple<RVVVector, 2>;
using vint16m2x3_t = VectorTuple<RVVVector, 3>;
using vint16m2x4_t = VectorTuple<RVVVector, 4>;
using vint16m4x2_t = VectorTuple<RVVVector, 2>;
using vint32mf2x2_t = VectorTuple<RVVVector, 2>;
using vint32mf2x3_t = VectorTuple<RVVVector, 3>;
using vint32mf2x4_t = VectorTuple<RVVVector, 4>;
using vint32mf2x5_t = VectorTuple<RVVVector, 5>;
using vint32mf2x6_t = VectorTuple<RVVVector, 6>;
using vint32mf2x7_t = VectorTuple<RVVVector, 7>;
using vint32mf2x8_t = VectorTuple<RVVVector, 8>;
using vint32m1x2_t = VectorTuple<RVVVector, 2>;
using vint32m1x3_t = VectorTuple<RVVVector, 3>;
using vint32m1x4_t = VectorTuple<RVVVector, 4>;
using vint32m1x5_t = VectorTuple<RVVVector, 5>;
using vint32m1x6_t = VectorTuple<RVVVector, 6>;
using vint32m1x7_t = VectorTuple<RVVVector, 7>;
using vint32m1x8_t = VectorTuple<RVVVector, 8>;
using vint32m2x2_t = VectorTuple<RVVVector, 2>;
using vint32m2x3_t = VectorTuple<RVVVector, 3>;
using vint32m2x4_t = VectorTuple<RVVVector, 4>;
using vint32m4x2_t = VectorTuple<RVVVector, 2>;
using vint64m1x2_t = VectorTuple<RVVVector, 2>;
using vint64m1x3_t = VectorTuple<RVVVector, 3>;
using vint64m1x4_t = VectorTuple<RVVVector, 4>;
using vint64m1x5_t = VectorTuple<RVVVector, 5>;
using vint64m1x6_t = VectorTuple<RVVVector, 6>;
using vint64m1x7_t = VectorTuple<RVVVector, 7>;
using vint64m1x8_t = VectorTuple<RVVVector, 8>;
using vint64m2x2_t = VectorTuple<RVVVector, 2>;
using vint64m2x3_t = VectorTuple<RVVVector, 3>;
using vint64m2x4_t = VectorTuple<RVVVector, 4>;
using vint64m4x2_t = VectorTuple<RVVVector, 2>;
using vuint8mf8x2_t = VectorTuple<RVVVector, 2>;
using vuint8mf8x3_t = VectorTuple<RVVVector, 3>;
using vuint8mf8x4_t = VectorTuple<RVVVector, 4>;
using vuint8mf8x5_t = VectorTuple<RVVVector, 5>;
using vuint8mf8x6_t = VectorTuple<RVVVector, 6>;
using vuint8mf8x7_t = VectorTuple<RVVVector, 7>;
using vuint8mf8x8_t = VectorTuple<RVVVector, 8>;
using vuint8mf4x2_t = VectorTuple<RVVVector, 2>;
using vuint8mf4x3_t = VectorTuple<RVVVector, 3>;
using vuint8mf4x4_t = VectorTuple<RVVVector, 4>;
using vuint8mf4x5_t = VectorTuple<RVVVector, 5>;
using vuint8mf4x6_t = VectorTuple<RVVVector, 6>;
using vuint8mf4x7_t = VectorTuple<RVVVector, 7>;
using vuint8mf4x8_t = VectorTuple<RVVVector, 8>;
using vuint8mf2x2_t = VectorTuple<RVVVector, 2>;
using vuint8mf2x3_t = VectorTuple<RVVVector, 3>;
using vuint8mf2x4_t = VectorTuple<RVVVector, 4>;
using vuint8mf2x5_t = VectorTuple<RVVVector, 5>;
using vuint8mf2x6_t = VectorTuple<RVVVector, 6>;
using vuint8mf2x7_t = VectorTuple<RVVVector, 7>;
using vuint8mf2x8_t = VectorTuple<RVVVector, 8>;
using vuint8m1x2_t = VectorTuple<RVVVector, 2>;
using vuint8m1x3_t = VectorTuple<RVVVector, 3>;
using vuint8m1x4_t = VectorTuple<RVVVector, 4>;
using vuint8m1x5_t = VectorTuple<RVVVector, 5>;
using vuint8m1x6_t = VectorTuple<RVVVector, 6>;
using vuint8m1x7_t = VectorTuple<RVVVector, 7>;
using vuint8m1x8_t = VectorTuple<RVVVector, 8>;
using vuint8m2x2_t = VectorTuple<RVVVector, 2>;
using vuint8m2x3_t = VectorTuple<RVVVector, 3>;
using vuint8m2x4_t = VectorTuple<RVVVector, 4>;
using vuint8m4x2_t = VectorTuple<RVVVector, 2>;
using vuint16mf4x2_t = VectorTuple<RVVVector, 2>;
using vuint16mf4x3_t = VectorTuple<RVVVector, 3>;
using vuint16mf4x4_t = VectorTuple<RVVVector, 4>;
using vuint16mf4x5_t = VectorTuple<RVVVector, 5>;
using vuint16mf4x6_t = VectorTuple<RVVVector, 6>;
using vuint16mf4x7_t = VectorTuple<RVVVector, 7>;
using vuint16mf4x8_t = VectorTuple<RVVVector, 8>;
using vuint16mf2x2_t = VectorTuple<RVVVector, 2>;
using vuint16mf2x3_t = VectorTuple<RVVVector, 3>;
using vuint16mf2x4_t = VectorTuple<RVVVector, 4>;
using vuint16mf2x5_t = VectorTuple<RVVVector, 5>;
using vuint16mf2x6_t = VectorTuple<RVVVector, 6>;
using vuint16mf2x7_t = VectorTuple<RVVVector, 7>;
using vuint16mf2x8_t = VectorTuple<RVVVector, 8>;
using vuint16m1x2_t = VectorTuple<RVVVector, 2>;
using vuint16m1x3_t = VectorTuple<RVVVector, 3>;
using vuint16m1x4_t = VectorTuple<RVVVector, 4>;
using vuint16m1x5_t = VectorTuple<RVVVector, 5>;
using vuint16m1x6_t = VectorTuple<RVVVector, 6>;
using vuint16m1x7_t = VectorTuple<RVVVector, 7>;
using vuint16m1x8_t = VectorTuple<RVVVector, 8>;
using vuint16m2x2_t = VectorTuple<RVVVector, 2>;
using vuint16m2x3_t = VectorTuple<RVVVector, 3>;
using vuint16m2x4_t = VectorTuple<RVVVector, 4>;
using vuint16m4x2_t = VectorTuple<RVVVector, 2>;
using vuint32mf2x3_t = VectorTuple<RVVVector, 3>;
using vuint32mf2x4_t = VectorTuple<RVVVector, 4>;
using vuint32mf2x5_t = VectorTuple<RVVVector, 5>;
using vuint32mf2x6_t = VectorTuple<RVVVector, 6>;
using vuint32mf2x7_t = VectorTuple<RVVVector, 7>;
using vuint32mf2x8_t = VectorTuple<RVVVector, 8>;
using vuint32m1x3_t = VectorTuple<RVVVector, 3>;
using vuint32m1x4_t = VectorTuple<RVVVector, 4>;
using vuint32m1x5_t = VectorTuple<RVVVector, 5>;
using vuint32m1x6_t = VectorTuple<RVVVector, 6>;
using vuint32m1x7_t = VectorTuple<RVVVector, 7>;
using vuint32m1x8_t = VectorTuple<RVVVector, 8>;
using vuint64m1x2_t = VectorTuple<RVVVector, 2>;
using vuint64m1x3_t = VectorTuple<RVVVector, 3>;
using vuint64m1x4_t = VectorTuple<RVVVector, 4>;
using vuint64m1x5_t = VectorTuple<RVVVector, 5>;
using vuint64m1x6_t = VectorTuple<RVVVector, 6>;
using vuint64m1x7_t = VectorTuple<RVVVector, 7>;
using vuint64m1x8_t = VectorTuple<RVVVector, 8>;
using vuint64m2x2_t = VectorTuple<RVVVector, 2>;
using vuint64m2x3_t = VectorTuple<RVVVector, 3>;
using vuint64m2x4_t = VectorTuple<RVVVector, 4>;
using vuint64m4x2_t = VectorTuple<RVVVector, 2>;
using vfloat16mf4x2_t = VectorTuple<RVVVector, 2>;
using vfloat16mf4x3_t = VectorTuple<RVVVector, 3>;
using vfloat16mf4x4_t = VectorTuple<RVVVector, 4>;
using vfloat16mf4x5_t = VectorTuple<RVVVector, 5>;
using vfloat16mf4x6_t = VectorTuple<RVVVector, 6>;
using vfloat16mf4x7_t = VectorTuple<RVVVector, 7>;
using vfloat16mf4x8_t = VectorTuple<RVVVector, 8>;
using vfloat16mf2x2_t = VectorTuple<RVVVector, 2>;
using vfloat16mf2x3_t = VectorTuple<RVVVector, 3>;
using vfloat16mf2x4_t = VectorTuple<RVVVector, 4>;
using vfloat16mf2x5_t = VectorTuple<RVVVector, 5>;
using vfloat16mf2x6_t = VectorTuple<RVVVector, 6>;
using vfloat16mf2x7_t = VectorTuple<RVVVector, 7>;
using vfloat16mf2x8_t = VectorTuple<RVVVector, 8>;
using vfloat16m1x4_t = VectorTuple<RVVVector, 4>;
using vfloat16m1x5_t = VectorTuple<RVVVector, 5>;
using vfloat16m1x6_t = VectorTuple<RVVVector, 6>;
using vfloat16m1x7_t = VectorTuple<RVVVector, 7>;
using vfloat16m1x8_t = VectorTuple<RVVVector, 8>;
using vfloat16m2x2_t = VectorTuple<RVVVector, 2>;
using vfloat16m2x3_t = VectorTuple<RVVVector, 3>;
using vfloat16m2x4_t = VectorTuple<RVVVector, 4>;
using vfloat16m4x2_t = VectorTuple<RVVVector, 2>;
using vfloat32mf2x2_t = VectorTuple<RVVVector, 2>;
using vfloat32mf2x3_t = VectorTuple<RVVVector, 3>;
using vfloat32mf2x4_t = VectorTuple<RVVVector, 4>;
using vfloat32mf2x5_t = VectorTuple<RVVVector, 5>;
using vfloat32mf2x6_t = VectorTuple<RVVVector, 6>;
using vfloat32mf2x7_t = VectorTuple<RVVVector, 7>;
using vfloat32mf2x8_t = VectorTuple<RVVVector, 8>;
using vfloat32m2x2_t = VectorTuple<RVVVector, 2>;
using vfloat32m2x3_t = VectorTuple<RVVVector, 3>;
using vfloat32m2x4_t = VectorTuple<RVVVector, 4>;
using vfloat32m4x2_t = VectorTuple<RVVVector, 2>;
using vfloat64m1x2_t = VectorTuple<RVVVector, 2>;
using vfloat64m1x3_t = VectorTuple<RVVVector, 3>;
using vfloat64m1x4_t = VectorTuple<RVVVector, 4>;
using vfloat64m1x5_t = VectorTuple<RVVVector, 5>;
using vfloat64m1x6_t = VectorTuple<RVVVector, 6>;
using vfloat64m1x7_t = VectorTuple<RVVVector, 7>;
using vfloat64m1x8_t = VectorTuple<RVVVector, 8>;
using vfloat64m2x2_t = VectorTuple<RVVVector, 2>;
using vfloat64m2x3_t = VectorTuple<RVVVector, 3>;
using vfloat64m2x4_t = VectorTuple<RVVVector, 4>;
using vfloat64m4x2_t = VectorTuple<RVVVector, 2>;

using vbool4_t  = MaskVector;
using vbool8_t  = MaskVector;
using vbool32_t = MaskVector;

// ===========================================================================
// VSETVL — returns min(avl, vlmax) where vlmax = VLEN/SEW * LMUL.
// Clamps vlmax to ≥1 to defend against pathologically small `vlen` values
// passed by the orchestrator: vsetvl returning 0 silently breaks any inner
// stripmine loop (`while (k > 0) k -= vl * SEW;` becomes infinite).
// ===========================================================================
inline size_t _rvv_vsetvl(size_t avl, size_t sew, size_t lmul) {
    return std::min(avl, std::max((size_t)1, g_ctx->vlen / sew * lmul));
}
} // namespace salt_cvc5
