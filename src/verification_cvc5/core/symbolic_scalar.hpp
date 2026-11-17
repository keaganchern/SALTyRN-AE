#pragma once
// SymbolicScalar<T> — typed wrapper carrying a cvc5::Term through C arithmetic.
// cvc5 port — mirrors src/verification/core/symbolic_scalar.hpp.
//
// Dispatches to BV or FP operations based on the C type via SymTraits<T>.
// Has NO implicit conversion to T — a missing intrinsic overload is a
// compile error, not a silent collapse to zero.

#include "context.hpp"
#include "symbolic_buffer.hpp"
#include "native_guard.hpp"

#include <cvc5/cvc5.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

// Forward-declare xnn_float16 (defined in xnn_minimal.h, included later by salt.hpp).
struct xnn_float16;

namespace salt_cvc5 {

// ---------------------------------------------------------------------------
// SymTraits<T> — maps C types to cvc5 sort kind and parameters
// ---------------------------------------------------------------------------
template<typename T> struct SymTraits;

template<> struct SymTraits<int8_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 8;
    static constexpr bool is_signed = true;
};
template<> struct SymTraits<uint8_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 8;
    static constexpr bool is_signed = false;
};
template<> struct SymTraits<int16_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 16;
    static constexpr bool is_signed = true;
};
template<> struct SymTraits<uint16_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 16;
    static constexpr bool is_signed = false;
};
template<> struct SymTraits<int32_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 32;
    static constexpr bool is_signed = true;
};
template<> struct SymTraits<uint32_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 32;
    static constexpr bool is_signed = false;
};
template<> struct SymTraits<int64_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 64;
    static constexpr bool is_signed = true;
};
template<> struct SymTraits<uint64_t> {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = 64;
    static constexpr bool is_signed = false;
};
// size_t and long where they are not the fixed-width types above (macOS: size_t
// is unsigned long, uint64_t unsigned long long). The placeholder keeps the
// specialization from repeating uint64_t's where they are the same type.
template<typename T> struct SymIntTraits {
    static constexpr bool is_fp = false;
    static constexpr size_t bits = sizeof(T) * 8;
    static constexpr bool is_signed = std::is_signed_v<T>;
};
struct SymUnusedUL; struct SymUnusedL;
template<> struct SymTraits<std::conditional_t<std::is_same_v<unsigned long, uint64_t>, SymUnusedUL, unsigned long>>
    : SymIntTraits<unsigned long> {};
template<> struct SymTraits<std::conditional_t<std::is_same_v<long, int64_t>, SymUnusedL, long>>
    : SymIntTraits<long> {};
template<> struct SymTraits<float> {
    static constexpr bool is_fp = true;
    static constexpr size_t bits = 32;
    static constexpr bool is_signed = true;
    static Sort sort(TermManager& tm) { return tm.mkFloatingPointSort(8, 24); }
};
template<> struct SymTraits<xnn_float16> {
    static constexpr bool is_fp = true;
    static constexpr size_t bits = 16;
    static constexpr bool is_signed = true;
    static Sort sort(TermManager& tm) { return tm.mkFloatingPointSort(5, 11); }
};

// ---------------------------------------------------------------------------
// Helper: float → IEEE 754 bit pattern → cvc5 FP value
// (cvc5 has no `mk_fp_value(sort, rm, decimal_string)` like bitwuzla — instead
// we go through the BV bit pattern.)
// ---------------------------------------------------------------------------
inline Term mk_fp32_from_float_pattern(TermManager& tm, float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    Term bv = tm.mkBitVector(32, static_cast<uint64_t>(bits));
    Op op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {8, 24});
    return tm.mkTerm(op, {bv});
}

inline Term mk_fp16_from_bits(TermManager& tm, uint16_t bits) {
    Term bv = tm.mkBitVector(16, static_cast<uint64_t>(bits));
    Op op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
    return tm.mkTerm(op, {bv});
}

// ---------------------------------------------------------------------------
// SymbolicScalar<T>
// ---------------------------------------------------------------------------
template<typename T>
class SymbolicScalar {
    Term term_;
    using Tr = SymTraits<T>;

public:
    // From symbolic buffer read at byte offset
    SymbolicScalar(SymbolicBuffer& buf, size_t byte_offset) {
        if constexpr (Tr::is_fp) {
            Term bv = buf.loadScalar(byte_offset, Tr::bits);
            if constexpr (Tr::bits == 32) {
                Op op = g_ctx->tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {8, 24});
                term_ = g_ctx->tm.mkTerm(op, {bv});
            } else {
                Op op = g_ctx->tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
                term_ = g_ctx->tm.mkTerm(op, {bv});
            }
        } else {
            term_ = buf.loadScalar(byte_offset, Tr::bits);
        }
    }

    // From concrete value (creates constant term)
    SymbolicScalar(T val) {
        if constexpr (Tr::is_fp) {
            if constexpr (Tr::bits == 32) {
                // A native float may be a sentinel (a symbolic element read or extracted natively): decoded.
                term_ = salt_decode_fp32(g_ctx->tm, static_cast<float>(val));
            } else {
                // xnn_float16 carries an underlying uint16_t bit pattern.
                term_ = mk_fp16_from_bits(g_ctx->tm,
                    *reinterpret_cast<const uint16_t*>(&val));
            }
        } else {
            term_ = mk_bv_val(g_ctx->tm, Tr::bits, static_cast<int64_t>(val));
        }
    }

    // From raw Term
    explicit SymbolicScalar(Term t) : term_(std::move(t)) {}

    Term term() const { return term_; }

    // --- Cast to another type ---
    template<typename U>
    SymbolicScalar<U> cast() const {
        using TrU = SymTraits<U>;
        // To its own type: the identity. (A float-to-float to_fp is the identity on every value, NaN
        // included, but a term the reference never has: it hid the value from the UF-first matching.)
        if constexpr (std::is_same_v<T, U>) {
            return *this;
        }
        // FP → FP
        else if constexpr (Tr::is_fp && TrU::is_fp) {
            uint32_t exp_bits = (TrU::bits == 32) ? 8 : 5;
            uint32_t sig_bits = (TrU::bits == 32) ? 24 : 11;
            Op op = g_ctx->tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_FP, {exp_bits, sig_bits});
            return SymbolicScalar<U>(g_ctx->tm.mkTerm(op, {g_ctx->fp.rounding_mode, term_}));
        }
        // Int → FP
        else if constexpr (!Tr::is_fp && TrU::is_fp) {
            Kind k = Tr::is_signed ? Kind::FLOATINGPOINT_TO_FP_FROM_SBV
                                    : Kind::FLOATINGPOINT_TO_FP_FROM_UBV;
            uint32_t exp_bits = (TrU::bits == 32) ? 8 : 5;
            uint32_t sig_bits = (TrU::bits == 32) ? 24 : 11;
            Op op = g_ctx->tm.mkOp(k, {exp_bits, sig_bits});
            return SymbolicScalar<U>(g_ctx->tm.mkTerm(op, {g_ctx->fp.rounding_mode, term_}));
        }
        // FP → Int
        else if constexpr (Tr::is_fp && !TrU::is_fp) {
            Kind k = TrU::is_signed ? Kind::FLOATINGPOINT_TO_SBV : Kind::FLOATINGPOINT_TO_UBV;
            Op op = g_ctx->tm.mkOp(k, {static_cast<uint32_t>(TrU::bits)});
            return SymbolicScalar<U>(g_ctx->tm.mkTerm(op, {g_ctx->fp.rounding_mode, term_}));
        }
        // Int → Int
        else {
            constexpr size_t sb = Tr::bits, db = TrU::bits;
            if constexpr (db > sb) {
                Kind k = Tr::is_signed ? Kind::BITVECTOR_SIGN_EXTEND : Kind::BITVECTOR_ZERO_EXTEND;
                Op op = g_ctx->tm.mkOp(k, {static_cast<uint32_t>(db - sb)});
                return SymbolicScalar<U>(g_ctx->tm.mkTerm(op, {term_}));
            } else if constexpr (db < sb) {
                Op op = g_ctx->tm.mkOp(Kind::BITVECTOR_EXTRACT,
                    {static_cast<uint32_t>(db - 1), 0u});
                return SymbolicScalar<U>(g_ctx->tm.mkTerm(op, {term_}));
            } else {
                return SymbolicScalar<U>(term_);
            }
        }
    }

    // NO implicit operator T(); use concrete_zero() only for explicit C-compat dummies.
    T concrete_zero() const { return T(0); }
};

// ---------------------------------------------------------------------------
// Arithmetic as C evaluates it. Operands meet in C's usual arithmetic
// conversions, the type of their sum, so int8 * int8 is an int product; unary
// minus and a shift's left operand are promoted the same way. Constants of any
// arithmetic type mix in alike. (xnn_float16 has no C arithmetic: same type.)
// ---------------------------------------------------------------------------
template<typename T, typename U, typename = void> struct SymArith { using type = T; };
template<typename T, typename U>
struct SymArith<T, U, std::void_t<decltype(std::declval<T>() + std::declval<U>())>> {
    using type = decltype(std::declval<T>() + std::declval<U>());
};
template<typename R>
inline SymbolicScalar<R> sym_arith(Kind bv, Kind fp, const SymbolicScalar<R>& a, const SymbolicScalar<R>& b) {
    if constexpr (SymTraits<R>::is_fp)
        return SymbolicScalar<R>(g_ctx->tm.mkTerm(fp, {g_ctx->fp.rounding_mode, a.term(), b.term()}));
    else
        return SymbolicScalar<R>(g_ctx->tm.mkTerm(bv, {a.term(), b.term()}));
}
#define SALT_SYM_ARITH(OP, BVK, FPK, INT_ONLY)                                                      \
    template<typename T, typename U>                                                                \
    auto operator OP(SymbolicScalar<T> a, SymbolicScalar<U> b) {                                    \
        using R = typename SymArith<T, U>::type;                                                    \
        static_assert(!(INT_ONLY && SymTraits<R>::is_fp), "bitwise operator on a float");           \
        return sym_arith<R>(Kind::BVK, Kind::FPK, a.template cast<R>(), b.template cast<R>());      \
    }                                                                                               \
    template<typename T, typename C, typename = std::enable_if_t<std::is_arithmetic_v<C>>>          \
    auto operator OP(SymbolicScalar<T> a, C c) {                                                    \
        using R = typename SymArith<T, C>::type;                                                    \
        static_assert(!(INT_ONLY && SymTraits<R>::is_fp), "bitwise operator on a float");           \
        return sym_arith<R>(Kind::BVK, Kind::FPK, a.template cast<R>(), SymbolicScalar<R>(static_cast<R>(c))); \
    }                                                                                               \
    template<typename T, typename C, typename = std::enable_if_t<std::is_arithmetic_v<C>>>          \
    auto operator OP(C c, SymbolicScalar<T> a) {                                                    \
        using R = typename SymArith<C, T>::type;                                                    \
        static_assert(!(INT_ONLY && SymTraits<R>::is_fp), "bitwise operator on a float");           \
        return sym_arith<R>(Kind::BVK, Kind::FPK, SymbolicScalar<R>(static_cast<R>(c)), a.template cast<R>()); \
    }
SALT_SYM_ARITH(+, BITVECTOR_ADD,  FLOATINGPOINT_ADD,  false)
SALT_SYM_ARITH(-, BITVECTOR_SUB,  FLOATINGPOINT_SUB,  false)
SALT_SYM_ARITH(*, BITVECTOR_MULT, FLOATINGPOINT_MULT, false)
SALT_SYM_ARITH(&, BITVECTOR_AND,  BITVECTOR_AND,      true)
SALT_SYM_ARITH(|, BITVECTOR_OR,   BITVECTOR_OR,       true)
SALT_SYM_ARITH(^, BITVECTOR_XOR,  BITVECTOR_XOR,      true)
#undef SALT_SYM_ARITH
template<typename T>
auto operator-(SymbolicScalar<T> a) {
    using R = typename SymArith<T, T>::type;
    Term x = a.template cast<R>().term();
    return SymbolicScalar<R>(g_ctx->tm.mkTerm(SymTraits<R>::is_fp ? Kind::FLOATINGPOINT_NEG : Kind::BITVECTOR_NEG, {x}));
}
template<typename T, typename C, typename = std::enable_if_t<std::is_integral_v<C>>>
auto operator<<(SymbolicScalar<T> a, C n) {
    using R = typename SymArith<T, T>::type;
    static_assert(!SymTraits<R>::is_fp, "shift of a float");
    return SymbolicScalar<R>(g_ctx->tm.mkTerm(Kind::BITVECTOR_SHL,
        {a.template cast<R>().term(), mk_bv_val(g_ctx->tm, SymTraits<R>::bits, (int64_t)n)}));
}
template<typename T, typename C, typename = std::enable_if_t<std::is_integral_v<C>>>
auto operator>>(SymbolicScalar<T> a, C n) {
    using R = typename SymArith<T, T>::type;
    static_assert(!SymTraits<R>::is_fp, "shift of a float");
    return SymbolicScalar<R>(g_ctx->tm.mkTerm(SymTraits<R>::is_signed ? Kind::BITVECTOR_ASHR : Kind::BITVECTOR_LSHR,
        {a.template cast<R>().term(), mk_bv_val(g_ctx->tm, SymTraits<R>::bits, (int64_t)n)}));
}
// Shift by a symbolic amount (x << s, 1 << (s - 1)): the result has the left
// operand's promoted type; the amount is converted to its width.
template<typename T, typename U>
auto sym_shift(Term v, SymbolicScalar<U> n, bool left) {
    using R = typename SymArith<T, T>::type;
    static_assert(!SymTraits<R>::is_fp && !SymTraits<U>::is_fp, "shift of a float");
    Kind k = left ? Kind::BITVECTOR_SHL : SymTraits<R>::is_signed ? Kind::BITVECTOR_ASHR : Kind::BITVECTOR_LSHR;
    return SymbolicScalar<R>(g_ctx->tm.mkTerm(k, {v, n.template cast<R>().term()}));
}
template<typename T, typename U>
auto operator<<(SymbolicScalar<T> a, SymbolicScalar<U> n) { return sym_shift<T>(a.template cast<typename SymArith<T, T>::type>().term(), n, true); }
template<typename T, typename U>
auto operator>>(SymbolicScalar<T> a, SymbolicScalar<U> n) { return sym_shift<T>(a.template cast<typename SymArith<T, T>::type>().term(), n, false); }
template<typename C, typename U, typename = std::enable_if_t<std::is_integral_v<C>>>
auto operator<<(C c, SymbolicScalar<U> n) {
    using R = typename SymArith<C, C>::type;
    return sym_shift<C>(SymbolicScalar<R>(static_cast<R>(c)).term(), n, true);
}
template<typename C, typename U, typename = std::enable_if_t<std::is_integral_v<C>>>
auto operator>>(C c, SymbolicScalar<U> n) {
    using R = typename SymArith<C, C>::type;
    return sym_shift<C>(SymbolicScalar<R>(static_cast<R>(c)).term(), n, false);
}
// Compound assignment as C defines it: `x op= y` is `x = (T)(x op y)`, T the type of x (a rebound local:
// `bias += zp << shift`). Defined only where `x op y` is (no / or %: those stay compile errors, a refusal).
#define SALT_SYM_COMPOUND(OP)                                                                       \
    template<typename T, typename V>                                                                \
    auto operator OP##=(SymbolicScalar<T>& a, const V& b) -> decltype((a OP b), a) {                \
        a = (a OP b).template cast<T>();                                                            \
        return a;                                                                                   \
    }
SALT_SYM_COMPOUND(+)
SALT_SYM_COMPOUND(-)
SALT_SYM_COMPOUND(*)
SALT_SYM_COMPOUND(&)
SALT_SYM_COMPOUND(|)
SALT_SYM_COMPOUND(^)
SALT_SYM_COMPOUND(<<)
SALT_SYM_COMPOUND(>>)
#undef SALT_SYM_COMPOUND
// A C declaration `T x = e;` converts e to T; emit.py rebinds a local that holds
// a symbolic value as `auto x = salt_as<T>(e)`.
template<typename T, typename U>
SymbolicScalar<T> salt_as(SymbolicScalar<U> x) { return x.template cast<T>(); }
template<typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
T salt_as(U x) { return static_cast<T>(x); }

// ---------------------------------------------------------------------------
// Comparisons with a constant, as a C bool. A branch on a symbolic integer is
// taken only where the assumptions asserted so far (the param constraints)
// decide it: then every allowed value takes it, so following it is exact.
// One they leave open throws rather than pick a side. Operands meet in C's
// usual arithmetic conversions (decltype of their sum).
// ---------------------------------------------------------------------------
// A branch the assumptions leave open forks: the harness reruns the shape once
// per outcome (salt_forced_branches holds the outcomes taken so far on this
// path), asserting the outcome taken, so the paths partition the parameter
// space and the shape counts only if every path verifies.
struct UndecidedBranch : std::runtime_error { using std::runtime_error::runtime_error; };
inline std::vector<bool> salt_forced_branches;
inline size_t salt_forced_next = 0;
// The assertions sharing free constants with cond, transitively. The others are independent of cond: when the
// assertions are satisfiable they decide cond exactly when this part does (when they are not, this part may
// leave it open; the full set is asked next). A param test inside a kernel's loop meets a stack of the
// kernel's own arithmetic, none of it about the params: the cone is the param constraints.
inline std::vector<Term> sym_cone(const Term& cond, const std::vector<Term>& as) {
    auto consts = [](const Term& t, std::vector<uint64_t>& out) {
        std::vector<Term> st{t};
        std::unordered_set<uint64_t> seen;
        while (!st.empty()) {
            Term x = st.back(); st.pop_back();
            if (!seen.insert(x.getId()).second) continue;
            if (x.getKind() == Kind::CONSTANT) out.push_back(x.getId());
            for (const Term& c : x) st.push_back(c);
        }
    };
    std::vector<const std::vector<uint64_t>*> cs;
    cs.reserve(as.size());
    for (const Term& a : as) {
        auto [it, fresh] = g_ctx->sym_consts_.try_emplace(a.getId());
        if (fresh) consts(a, it->second);
        cs.push_back(&it->second);
    }
    std::vector<uint64_t> c0;
    consts(cond, c0);
    std::unordered_set<uint64_t> live(c0.begin(), c0.end());
    std::vector<bool> in(as.size(), false);
    std::vector<Term> cone;
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 0; i < as.size(); i++) {
            if (in[i] || std::none_of(cs[i]->begin(), cs[i]->end(), [&](uint64_t c) { return live.count(c) > 0; }))
                continue;
            in[i] = changed = true;
            cone.push_back(as[i]);
            live.insert(cs[i]->begin(), cs[i]->end());
        }
    }
    return cone;
}
inline std::optional<bool> sym_decide_under(const Term& cond, const std::vector<Term>& as) {
    for (bool want : {true, false}) {
        Solver s(g_ctx->tm);
        for (const Term& a : as) s.assertFormula(a);
        s.assertFormula(want ? g_ctx->tm.mkTerm(Kind::NOT, {cond}) : cond);
        if (s.checkSat().isUnsat()) return want;
    }
    return std::nullopt;
}
// The assertions without the kernels' definitions. A kernel's arithmetic reaches the solver as definitions of fresh
// symbols, (= x e) or (= (to_fp x) e) with x in no earlier assertion and not in e: whatever the other symbols are,
// x can be chosen to satisfy it. So a set of such definitions whose symbols occur in nothing that is kept (no kept
// assertion, not cond) is satisfiable on top of any model of the rest, and the rest decides cond exactly when the
// whole stack does. That turns a test of a parameter after the reference kernel ran (zero_point == 0 against N
// lanes of float arithmetic, seconds per test and per path) into a test against the parameter constraints.
inline std::vector<Term> sym_without_definitions(const Term& cond, const std::vector<Term>& as, size_t n) {
    auto consts = [](const Term& t, std::vector<uint64_t>& out) {
        std::vector<Term> st{t};
        std::unordered_set<uint64_t> seen;
        while (!st.empty()) {
            Term x = st.back(); st.pop_back();
            if (!seen.insert(x.getId()).second) continue;
            if (x.getKind() == Kind::CONSTANT) out.push_back(x.getId());
            for (const Term& c : x) st.push_back(c);
        }
    };
    std::vector<std::vector<uint64_t>> cs(as.size());
    std::vector<uint64_t> def(as.size(), 0);             // the symbol an assertion defines, 0 if it is kept
    std::unordered_set<uint64_t> earlier;
    for (size_t i = 0; i < as.size(); i++) {
        consts(as[i], cs[i]);
        if (i >= n && as[i].getKind() == Kind::EQUAL && as[i].getNumChildren() == 2) {
            for (int side = 0; side < 2 && !def[i]; side++) {
                Term x = as[i][side];
                if (x.getKind() == Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV) x = x[0];
                if (x.getKind() != Kind::CONSTANT || earlier.count(x.getId())) continue;
                std::vector<uint64_t> other;
                consts(as[i][1 - side], other);
                if (std::find(other.begin(), other.end(), x.getId()) == other.end()) def[i] = x.getId();
            }
        }
        earlier.insert(cs[i].begin(), cs[i].end());
    }
    std::vector<uint64_t> c0;
    consts(cond, c0);
    std::unordered_set<uint64_t> live(c0.begin(), c0.end());
    for (size_t i = 0; i < as.size(); i++)
        if (!def[i]) live.insert(cs[i].begin(), cs[i].end());
    for (bool changed = true; changed;) {                 // a definition whose symbol something kept uses is kept
        changed = false;
        for (size_t i = 0; i < as.size(); i++) {
            if (!def[i] || !live.count(def[i])) continue;
            def[i] = 0;
            live.insert(cs[i].begin(), cs[i].end());
            changed = true;
        }
    }
    std::vector<Term> kept;
    for (size_t i = 0; i < as.size(); i++)
        if (!def[i]) kept.push_back(as[i]);
    return kept;
}
inline bool sym_decided(const Term& cond) {
    if (auto c = g_ctx->sym_decided_.find(cond.getId()); c != g_ctx->sym_decided_.end()) return c->second;
    std::vector<Term> as = g_ctx->solver->getAssertions();
    auto t0 = std::chrono::steady_clock::now();
    // First the part of the harness's assumptions that shares symbols with cond, then every assertion (the
    // kernels' arithmetic too, as before): each a subset of the next, so a decision holds in every model.
    size_t n = std::min(g_ctx->assumptions_, as.size());
    std::vector<Term> cone = sym_cone(cond, std::vector<Term>(as.begin(), as.begin() + n));
    std::optional<bool> d = sym_decide_under(cond, cone);
    bool full = !d && cone.size() < as.size();
    if (full && !std::getenv("SALT_SYM_FULL")) {
        // ... but the kernels' definitions cannot decide it (see sym_without_definitions): ask the rest only.
        std::vector<Term> kept = sym_without_definitions(cond, as, n);
        if (kept.size() < as.size()) as = sym_cone(cond, kept);
    }
    if (full) d = sym_decide_under(cond, as);
    if (const char* e = std::getenv("SALT_SYM_DEBUG"); e && *e == '1')
        std::fprintf(stderr, "[sym] %s: %zu of %zu assertions (%zu assumptions)%s, %.2fs: %.120s\n",
                     d ? (*d ? "true" : "false") : "open", cone.size(), as.size(), n, full ? " then all" : "",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), cond.toString().c_str());
    if (d) {
        g_ctx->sym_decided_[cond.getId()] = *d;
        return *d;
    }
    if (salt_forced_next < salt_forced_branches.size()) {
        bool want = salt_forced_branches[salt_forced_next++];
        g_ctx->assert_(want ? cond : g_ctx->tm.mkTerm(Kind::NOT, {cond}));
        return want;
    }
    throw UndecidedBranch("branch on a symbolic value the param constraints do not decide: " +
                          cond.toString().substr(0, 200));
}
template<typename T, typename C>
bool sym_compare(SymbolicScalar<T> a, C c, Kind sk, Kind uk) {
    using CT = decltype(T{} + C{});
    Term x = a.template cast<CT>().term(), y = SymbolicScalar<CT>(static_cast<CT>(c)).term();
    return sym_decided(g_ctx->tm.mkTerm(SymTraits<CT>::is_signed ? sk : uk, {x, y}));
}
#define SALT_SYM_CMP(OP, SK, UK, RSK, RUK)                                                         \
    template<typename T, typename C, typename = std::enable_if_t<!SymTraits<T>::is_fp && std::is_integral_v<C>>> \
    bool operator OP(SymbolicScalar<T> a, C c) { return sym_compare(a, c, Kind::SK, Kind::UK); }      \
    template<typename T, typename C, typename = std::enable_if_t<!SymTraits<T>::is_fp && std::is_integral_v<C>>> \
    bool operator OP(C c, SymbolicScalar<T> a) { return sym_compare(a, c, Kind::RSK, Kind::RUK); }
// Two symbolic operands: the same rule, after the usual arithmetic conversions.
template<typename T, typename U>
bool sym_compare2(SymbolicScalar<T> a, SymbolicScalar<U> b, Kind sk, Kind uk) {
    using CT = typename SymArith<T, U>::type;
    return sym_decided(g_ctx->tm.mkTerm(SymTraits<CT>::is_signed ? sk : uk,
                                        {a.template cast<CT>().term(), b.template cast<CT>().term()}));
}
#define SALT_SYM_CMP2(OP, SK, UK)                                                              \
    template<typename T, typename U, typename = std::enable_if_t<!SymTraits<T>::is_fp && !SymTraits<U>::is_fp>> \
    bool operator OP(SymbolicScalar<T> a, SymbolicScalar<U> b) { return sym_compare2(a, b, Kind::SK, Kind::UK); }
SALT_SYM_CMP2(<,  BITVECTOR_SLT, BITVECTOR_ULT)
SALT_SYM_CMP2(<=, BITVECTOR_SLE, BITVECTOR_ULE)
SALT_SYM_CMP2(>,  BITVECTOR_SGT, BITVECTOR_UGT)
SALT_SYM_CMP2(>=, BITVECTOR_SGE, BITVECTOR_UGE)
SALT_SYM_CMP2(==, EQUAL, EQUAL)
SALT_SYM_CMP2(!=, DISTINCT, DISTINCT)
#undef SALT_SYM_CMP2
SALT_SYM_CMP(<,  BITVECTOR_SLT, BITVECTOR_ULT, BITVECTOR_SGT, BITVECTOR_UGT)
SALT_SYM_CMP(<=, BITVECTOR_SLE, BITVECTOR_ULE, BITVECTOR_SGE, BITVECTOR_UGE)
SALT_SYM_CMP(>,  BITVECTOR_SGT, BITVECTOR_UGT, BITVECTOR_SLT, BITVECTOR_ULT)
SALT_SYM_CMP(>=, BITVECTOR_SGE, BITVECTOR_UGE, BITVECTOR_SLE, BITVECTOR_ULE)
SALT_SYM_CMP(==, EQUAL, EQUAL, EQUAL, EQUAL)
SALT_SYM_CMP(!=, DISTINCT, DISTINCT, DISTINCT, DISTINCT)
#undef SALT_SYM_CMP

// ---------------------------------------------------------------------------
// Float comparisons as C evaluates them (IEEE 754): ==, <, <=, >, >= are false
// when an operand is NaN, != is the negation of ==, and +0 == -0. A branch on
// one is decided by the assumptions, or forks, exactly as for integers (so
// `vmax != vmax` is false under a "max is not NaN" constraint and the branch is
// never taken). Operands meet in C's usual conversions: an integer converts to
// float; a float meets a double in double, which is the float comparison only
// when the double is a float value, so any other double constant is refused.
// Return-type SFINAE keeps these apart from the integer templates above.
// ---------------------------------------------------------------------------
template<typename C>
inline Term sym_fp32_const(C c) {
    float f = static_cast<float>(c);
    if constexpr (std::is_floating_point_v<C>) {
        if (!(std::isnan(c) && std::isnan(f)) && static_cast<long double>(f) != static_cast<long double>(c))
            throw std::runtime_error("float compared with a constant float cannot represent (a double comparison)");
    }
    return SymbolicScalar<float>(f).term();
}
inline Term sym_fp_rel(Kind k, bool negate, const Term& x, const Term& y) {
    Term r = g_ctx->tm.mkTerm(k, {x, y});
    return negate ? g_ctx->tm.mkTerm(Kind::NOT, {r}) : r;
}
template<typename T> constexpr bool sym_is_f32 = std::is_same_v<T, float>;
#define SALT_SYM_FCMP(OP, K, RK, NEG)                                                                           \
    template<typename T, typename C>                                                                            \
    std::enable_if_t<sym_is_f32<T> && std::is_arithmetic_v<C>, bool> operator OP(SymbolicScalar<T> a, C c) {    \
        return sym_decided(sym_fp_rel(Kind::K, NEG, a.term(), sym_fp32_const(c)));                              \
    }                                                                                                           \
    template<typename T, typename C>                                                                            \
    std::enable_if_t<sym_is_f32<T> && std::is_arithmetic_v<C>, bool> operator OP(C c, SymbolicScalar<T> a) {    \
        return sym_decided(sym_fp_rel(Kind::RK, NEG, a.term(), sym_fp32_const(c)));                             \
    }                                                                                                           \
    template<typename T, typename U>                                                                            \
    std::enable_if_t<(sym_is_f32<T> || sym_is_f32<U>) &&                                                      \
                     (sym_is_f32<T> || !SymTraits<T>::is_fp) && (sym_is_f32<U> || !SymTraits<U>::is_fp), bool>  \
    operator OP(SymbolicScalar<T> a, SymbolicScalar<U> b) {                                                     \
        return sym_decided(sym_fp_rel(Kind::K, NEG, a.template cast<float>().term(), b.template cast<float>().term())); \
    }
SALT_SYM_FCMP(==, FLOATINGPOINT_EQ,  FLOATINGPOINT_EQ,  false)
SALT_SYM_FCMP(!=, FLOATINGPOINT_EQ,  FLOATINGPOINT_EQ,  true)
SALT_SYM_FCMP(<,  FLOATINGPOINT_LT,  FLOATINGPOINT_GT,  false)
SALT_SYM_FCMP(<=, FLOATINGPOINT_LEQ, FLOATINGPOINT_GEQ, false)
SALT_SYM_FCMP(>,  FLOATINGPOINT_GT,  FLOATINGPOINT_LT,  false)
SALT_SYM_FCMP(>=, FLOATINGPOINT_GEQ, FLOATINGPOINT_LEQ, false)
#undef SALT_SYM_FCMP

} // namespace salt_cvc5
