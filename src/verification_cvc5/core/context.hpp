#pragma once
// VerificationContext for the cvc5 port — mirrors src/verification/core/context.hpp
// but uses the cvc5 C++ API.  All identifiers live in `salt_cvc5::` to avoid
// any chance of ODR collisions with the bitwuzla tree even in the unlikely
// event both are linked together.

#include <cvc5/cvc5.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>   // unlink (external-solver temp file)

// salt_float16 — same shim as the bitwuzla tree (option (a) from the plan: a fresh
// struct in this tree, completely independent from the one in src/verification).
// Layout-compatible with xnn_float16 (struct path, XNN_HAVE_FLOAT16=0).
struct salt_float16 {
    uint16_t value;
    salt_float16() = default;
    explicit constexpr salt_float16(uint16_t v) : value(v) {}
    // Implicit xnn_float16 → salt_float16 — required because kernels (e.g. f16-vmulcaddc)
    // pass xnn_float16 scalar args to _vf_f16 intrinsics.  Layout-compat is asserted
    // in salt.hpp.  Non-explicit on purpose: makes the conversion implicit.
    salt_float16(xnn_float16 v) : value(v.value) {}
    salt_float16(float f) {
        uint32_t x;
        __builtin_memcpy(&x, &f, sizeof(x));
        uint16_t sign = static_cast<uint16_t>((x >> 31) << 15);
        int32_t  exp32 = static_cast<int32_t>((x >> 23) & 0xFFu);
        uint32_t mant  = x & 0x7FFFFFu;
        if (exp32 == 0xFF) {
            value = sign | 0x7C00u | (mant ? 0x0200u : 0);
            return;
        }
        int32_t exp16 = exp32 - 127 + 15;
        if (exp16 >= 31) { value = sign | 0x7C00u; return; }
        if (exp16 <= 0) {
            if (exp16 < -10) { value = sign; return; }
            mant |= 0x800000u;
            value = sign | static_cast<uint16_t>(mant >> (14 - exp16));
            return;
        }
        value = sign | static_cast<uint16_t>(exp16 << 10)
              | static_cast<uint16_t>(mant >> 13);
    }
    // See symbolic_buffer.hpp for the out-of-line copy-ctor / dtor that
    // implement the scalar-Term side channel.
    salt_float16(const salt_float16& other);
    salt_float16& operator=(const salt_float16& other);
    ~salt_float16();
};

// Compat: where the compiler has NO native _Float16 (e.g. GCC ≤11 on x86-64),
// let inlined kernel code that names `_Float16` resolve to our symbolic struct.
// Where _Float16 IS native (GCC ≥12), kernels use the native scalar — f16 kernel
// support is Phase-3 — but the shim itself uses salt_float16 and is unaffected.
#ifndef __FLT16_MAX__
typedef salt_float16 _Float16;
#endif

namespace salt_cvc5 {

using cvc5::Term;
using cvc5::Sort;
using cvc5::Op;
using cvc5::Kind;
using cvc5::TermManager;
using cvc5::Solver;
using cvc5::Result;
using cvc5::RoundingMode;

// ---------------------------------------------------------------------------
// FP profile — frozen AArch64 FP environment
// ---------------------------------------------------------------------------
struct FPProfile {
    Term rounding_mode;
    bool flush_to_zero    = false;
    bool flush_to_zero_f16= false;
    bool default_nan      = false;
    bool ah               = false;
};

// ---------------------------------------------------------------------------
// Element kind — for the comparison helper
// ---------------------------------------------------------------------------
enum class ElementKind {
    SINT, UINT, F16, BF16, F32, F64, MASK,
};

class SymbolicBuffer; // forward decl

// ---------------------------------------------------------------------------
// VerificationContext
// ---------------------------------------------------------------------------
// Backend-agnostic check_sat() result.  Mirrors the bitwuzla backend's enum
// so generated harness code is portable across backends.
enum class CheckResult { Sat, Unsat, Unknown };

struct VerificationContext {
    TermManager tm;
    std::unique_ptr<Solver> solver;
    FPProfile fp;
    size_t vlen;

    std::map<std::string, std::unique_ptr<SymbolicBuffer>> buffers;
    std::vector<SymbolicBuffer*> registered_buffers;

    // Scalar reduction result (rsum-style `*output += vmv_x_s/vget_lane(...)`):
    // the scalar-extract shims stash the symbolic sum here so the harness can
    // compare it directly — the scalar `*output +=` store bypasses the symbolic
    // output buffer.  The harness clears this around each kernel call.
    Term scalar_result_;
    bool has_scalar_result_ = false;
    // A reduction kernel's `*out += sum` is compared through this side channel, so
    // the harness checks that a kernel captured exactly one sum (scalar_result_count).
    size_t scalar_result_count_ = 0;
    void set_scalar_result(const Term& t) { scalar_result_ = t; has_scalar_result_ = true; scalar_result_count_++; }
    void clear_scalar_result() { has_scalar_result_ = false; scalar_result_count_ = 0; }
    size_t scalar_result_count() const { return scalar_result_count_; }
    bool has_scalar_result() const { return has_scalar_result_; }
    Term scalar_result() const { return scalar_result_; }
    // Floats extracted from a vector lane (vget_lane_f32, __riscv_vfmv_f_s): the extract returns a sentinel naming
    // an entry here (buffer id 0, see kScalarRefSentinel), so a native copy handed back to an intrinsic decodes to
    // the lane's 32-bit term, and native arithmetic on it is refused like on any sentinel (native_guard.hpp).
    std::vector<Term> extracted_fp_;

    // UF-abstraction (§9c.E): SALT_UF_ABSTRACT=mul,fma,... replaces those FP ops
    // with a shared uninterpreted function on BOTH kernels (the rewrite is applied
    // to the asserted formula, so it's symmetric by construction).  Sound but
    // incomplete: UNSAT-under-abstraction ⇒ concrete equivalence; SAT may be a
    // spurious artifact of an op the abstraction over-generalized.
    std::string uf_ops_;
    std::unordered_map<std::string, Term> uf_syms_;   // "op@sort" → function symbol
    // UF-first (SALT_UF_FIRST=mul,fma,add,sub[,bvmul]): check() first solves a copy
    // of the whole assertion stack with those ops abstracted, and accepts only UNSAT.
    // Sound both ways: the real ops are one interpretation of the UFs, so UNSAT
    // there means UNSAT here; any other answer falls through to the exact query.
    // bvmul abstracts integer products of two non-constant operands (a symbolic
    // parameter times data), each bounded by its operands' ranges.
    std::string uf_first_ops_;
    const char* method_ = "smt";   // how the last Unsat was obtained: "uf" or "smt"
    // Branch conditions the assertions decide (sym_decided): assertions are only ever added in a context,
    // so a decided condition stays decided; the same test in a loop is solved once.
    std::unordered_map<uint64_t, bool> sym_decided_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> sym_consts_;   // free constants per assertion
    // The assertions made before the reference kernel runs are the harness's assumptions (input domains,
    // parameter constraints); the harness marks the point. sym_decided asks them first: a subset of the
    // assertions, so what they decide every model decides, without the kernels' arithmetic.
    size_t assumptions_ = SIZE_MAX;
    void mark_assumptions() { assumptions_ = solver->getAssertions().size(); }
    size_t uf_axioms_ = 0;         // interval axioms added in the last UF-first stage

    explicit VerificationContext(size_t vlen_) : vlen(vlen_) {
        solver = std::make_unique<Solver>(tm);
        solver->setOption("produce-models", "true");
        // fp16 (5/11) is only supported under the experimental FP solver in
        // cvc5 1.3.x.  Enable unconditionally — kernels that don't touch fp16
        // are unaffected.
        solver->setOption("fp-exp", "true");
        // Opt-in stats: SALT_STATS=1 enables cvc5 internal counters and
        // print_stats() dumps them.  Off by default — has measurable overhead.
        if (const char* s = std::getenv("SALT_STATS"); s && std::string(s) == "1") {
            solver->setOption("stats", "true");
            solver->setOption("stats-internal", "true");
        }
        // Opt-in solver-option override: SALT_CVC5_OPTIONS="key1=val1,key2=val2"
        // applies arbitrary cvc5 setOption calls before any assertions, so the
        // user can A/B test --bv-solver=bitblast etc. without rebuilding.
        if (const char* opts = std::getenv("SALT_CVC5_OPTIONS"); opts && *opts) {
            std::string s = opts;
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string kv = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                size_t eq = kv.find('=');
                if (eq != std::string::npos) {
                    std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
                    try {
                        solver->setOption(k, v);
                        std::fprintf(stderr, "[SALT_CVC5_OPTIONS] set %s=%s\n", k.c_str(), v.c_str());
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "[SALT_CVC5_OPTIONS] FAILED %s=%s: %s\n", k.c_str(), v.c_str(), e.what());
                    }
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            std::fflush(stderr);
        }
        if (const char* u = std::getenv("SALT_UF_FIRST"); u && *u) uf_first_ops_ = u;
        if (const char* u = std::getenv("SALT_UF_ABSTRACT"); u && *u) {
            uf_ops_ = u;
            std::fprintf(stderr, "[SALT_UF_ABSTRACT] abstracting FP ops: %s\n", u);
            std::fflush(stderr);
        }
        fp.rounding_mode = tm.mkRoundingMode(RoundingMode::ROUND_NEAREST_TIES_TO_EVEN);
        // Drop any stale entries left over from a previous fixture's
        // VerificationContext.  Without this, salt_float16's copy-ctor (and any
        // future scalar side-channel) can re-use a Term tied to a destroyed
        // TermManager, triggering "term not associated with this term manager".
        // Definition is later in this file — call via the helper.
        clear_scalar_terms();
    }

    // Clear the Term side-channels (g_scalar_terms, g_fp_bv_cache) BEFORE the TermManager
    // member is torn down — otherwise the thread_local maps outlive `tm` and their cached
    // Terms' destructors run against a freed TermManager (use-after-free / SIGSEGV at exit).
    ~VerificationContext();

    // Forward-declared helper; definition follows g_scalar_terms below.
    static void clear_scalar_terms();

    void print_stats(FILE* out = stderr) {
        if (const char* s = std::getenv("SALT_STATS"); !s || std::string(s) != "1") return;
        std::fprintf(out, "=== cvc5 statistics ===\n");
        auto stats = solver->getStatistics();
        std::fprintf(out, "%s", stats.toString().c_str());
        std::fprintf(out, "=== end stats ===\n");
        std::fflush(out);
    }

    // A counterexample as JSON, for an independent concrete replay: every registered buffer's bytes (the model's
    // values of the symbolic ones) and the shape. Written only when SALT_CEX_JSON names a directory, after a
    // counterexample was found (<dir>/cex-<shape>.json); it does not change the check.
    void dump_model_json(const std::string& shape);

    SymbolicBuffer& registerBuffer(const std::string& name, size_t num_bytes);
    // Read-only concrete buffer: loads return BV constants synthesized from
    // `data`; no symbolic byte vars are created.  Stores throw — kernels must
    // not write to a buffer registered concretely.  Replaces the
    // registerBuffer + per-byte assert_eq idiom used for w_ptr/params/LUTs.
    SymbolicBuffer& registerConcreteBuffer(const std::string& name,
                                            const void* data, size_t num_bytes);
    // Remove a buffer registered for a scope (a kernel's local scratch array): no longer found by address.
    void unregisterBuffer(SymbolicBuffer* b);
    SymbolicBuffer& findBuffer(const void* ptr);
    SymbolicBuffer* findBufferSafe(const void* ptr) noexcept;
    SymbolicBuffer* findBufferById(uint8_t id) noexcept;

    Sort bv_sort(size_t bits) { return tm.mkBitVectorSort(static_cast<uint32_t>(bits)); }
    Sort fp32_sort() { return tm.mkFloatingPointSort(8, 24); }
    Sort fp16_sort() { return tm.mkFloatingPointSort(5, 11); }

    // -----------------------------------------------------------------
    // Backend-agnostic helper API — mirrors src/verification/core/context.hpp
    // so generated harness code is portable across backends.  Emitted harness
    // code MUST go through these helpers; never call `tm.mk*` or `solver->`
    // methods directly from generated code.
    // -----------------------------------------------------------------

    // BV constructors
    Term bv_val(size_t bits, uint64_t val) {
        return tm.mkBitVector(static_cast<uint32_t>(bits), val);
    }
    Term bv_const(size_t bits, const std::string& name) {
        return tm.mkConst(bv_sort(bits), name);
    }

    // Logical / equality
    Term equal(Term a, Term b) { return tm.mkTerm(Kind::EQUAL, {a, b}); }
    Term add(Term a, Term b)   { return tm.mkTerm(Kind::BITVECTOR_ADD, {a, b}); }
    Term land_(Term a, Term b) { return tm.mkTerm(Kind::AND,   {a, b}); }
    Term lor_ (Term a, Term b) { return tm.mkTerm(Kind::OR,    {a, b}); }
    Term lnot_(Term t)         { return tm.mkTerm(Kind::NOT,   {t});    }
    Term ite  (Term c, Term t, Term e) { return tm.mkTerm(Kind::ITE, {c, t, e}); }

    // BV comparisons (for symbolic-param validity / range constraints)
    Term bv_sgt(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_SGT, {a, b}); }
    Term bv_sge(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_SGE, {a, b}); }
    Term bv_slt(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_SLT, {a, b}); }
    Term bv_sle(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_SLE, {a, b}); }
    Term bv_ugt(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_UGT, {a, b}); }
    Term bv_uge(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_UGE, {a, b}); }
    Term bv_ult(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_ULT, {a, b}); }
    Term bv_ule(Term a, Term b) { return tm.mkTerm(Kind::BITVECTOR_ULE, {a, b}); }

    // FP comparisons (for param range constraints)
    Term fp16_geq(Term a, Term b) { return tm.mkTerm(Kind::FLOATINGPOINT_GEQ, {a, b}); }
    Term fp16_leq(Term a, Term b) { return tm.mkTerm(Kind::FLOATINGPOINT_LEQ, {a, b}); }
    Term fp32_geq(Term a, Term b) { return tm.mkTerm(Kind::FLOATINGPOINT_GEQ, {a, b}); }
    Term fp32_leq(Term a, Term b) { return tm.mkTerm(Kind::FLOATINGPOINT_LEQ, {a, b}); }
    // Reinterpret a 16-bit BV as an IEEE-754 binary16 FP term.
    Term fp16_from_bv(Term bv16) {
        Op op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
        return tm.mkTerm(op, {bv16});
    }

    // -----------------------------------------------------------------
    // UF-abstraction rewrite (§9c.E)
    // -----------------------------------------------------------------
    bool uf_has(const char* op) const { return uf_ops_.find(op) != std::string::npos; }

    // Shared uninterpreted function symbol for (op, FP sort), arity-typed.  One
    // symbol per (op,sort) for the whole context ⇒ both kernels' occurrences are
    // the SAME function ⇒ congruence proves equivalence.
    Term uf_sym(const std::string& op, const Sort& s, size_t arity) {
        std::string key = op + "@" + s.toString();
        auto it = uf_syms_.find(key);
        if (it != uf_syms_.end()) return it->second;
        std::vector<Sort> dom(arity, s);
        Term f = tm.mkConst(tm.mkFunctionSort(dom, s),
                            "uf_" + op + "_" + std::to_string(uf_syms_.size()));
        uf_syms_[key] = f;
        return f;
    }

    Term uf_rewrite_rec(const Term& t, std::unordered_map<uint64_t, Term>& memo) {
        if (t.getNumChildren() == 0) return t;          // const / var / rounding mode
        auto it = memo.find(t.getId());
        if (it != memo.end()) return it->second;
        std::vector<Term> ch;
        ch.reserve(t.getNumChildren());
        for (const Term& c : t) ch.push_back(uf_rewrite_rec(c, memo));
        Kind k = t.getKind();
        Term out;
        // An integer converts to the same float however it was widened: drop a
        // sign extension under a signed conversion and any zero extension (which
        // makes it unsigned), so the abstracted products below see one argument
        // term where two kernels convert at different widths (vfcvt of an int32
        // vs vfwcvt of the int16 it extends).
        if ((k == Kind::FLOATINGPOINT_TO_FP_FROM_SBV || k == Kind::FLOATINGPOINT_TO_FP_FROM_UBV) &&
            ch.size() == 2) {
            bool sgn = k == Kind::FLOATINGPOINT_TO_FP_FROM_SBV;
            Term x = ch[1];
            for (;;) {
                if (sgn && x.getKind() == Kind::BITVECTOR_SIGN_EXTEND) x = x[0];
                else if (x.getKind() == Kind::BITVECTOR_ZERO_EXTEND &&
                         (!sgn || x.getOp()[0].getUInt32Value() > 0)) { x = x[0]; sgn = false; }
                else break;
            }
            if (x != ch[1]) {
                Op op = tm.mkOp(sgn ? Kind::FLOATINGPOINT_TO_FP_FROM_SBV : Kind::FLOATINGPOINT_TO_FP_FROM_UBV,
                                {t.getOp()[0].getUInt32Value(), t.getOp()[1].getUInt32Value()});
                out = tm.mkTerm(op, {ch[0], x});
                memo[t.getId()] = out;
                return out;
            }
        }
        // FP arith children are [rm, operands…]. One UF per (op, sort, rounding
        // mode). IEEE add, mul and fma's product are commutative, so operands are
        // ordered by id: a*b and b*a abstract to the same application.
        auto sym = [&](const char* op, size_t arity) {
            return uf_sym(std::string(op) + "@" + ch[0].toString(), t.getSort(), arity);
        };
        auto ord = [](Term& a, Term& b) { if (b.getId() < a.getId()) std::swap(a, b); };
        // A product with a constant zero factor stays interpreted: it is ±0 or NaN
        // whatever the other factor, so a kernel that skips it (an igemm row from
        // the zero buffer) still matches one that accumulates it, which a UF hides.
        auto zero_factor = [&] {
            double v;
            return (fp32_const(ch[1], v) && v == 0.0) || (fp32_const(ch[2], v) && v == 0.0);
        };
        if ((k == Kind::FLOATINGPOINT_MULT || k == Kind::FLOATINGPOINT_FMA) && ch.size() >= 3 && zero_factor())
            out = tm.mkTerm(k, ch);
        else if (k == Kind::FLOATINGPOINT_MULT && uf_has("mul")) {
            ord(ch[1], ch[2]);
            out = tm.mkTerm(Kind::APPLY_UF, {sym("mul", 2), ch[1], ch[2]});
        } else if (k == Kind::FLOATINGPOINT_FMA && uf_has("fma")) {
            ord(ch[1], ch[2]);
            out = tm.mkTerm(Kind::APPLY_UF, {sym("fma", 3), ch[1], ch[2], ch[3]});
        } else if (k == Kind::FLOATINGPOINT_ADD && uf_has("add")) {
            ord(ch[1], ch[2]);
            out = tm.mkTerm(Kind::APPLY_UF, {sym("add", 2), ch[1], ch[2]});
        } else if (k == Kind::FLOATINGPOINT_SUB && uf_has("sub"))
            out = tm.mkTerm(Kind::APPLY_UF, {sym("sub", 2), ch[1], ch[2]});
        else if (k == Kind::BITVECTOR_MULT && uf_has("bvmul") && ch.size() == 2 &&
                 ch[0].getKind() != Kind::CONST_BITVECTOR && ch[1].getKind() != Kind::CONST_BITVECTOR) {
            ord(ch[0], ch[1]);
            out = tm.mkTerm(Kind::APPLY_UF, {uf_sym("bvmul", t.getSort(), 2), ch[0], ch[1]});
        } else if (t.hasOp())
            out = tm.mkTerm(t.getOp(), ch);
        else
            out = tm.mkTerm(k, ch);
        memo[t.getId()] = out;
        return out;
    }

    Term uf_rewrite(const Term& formula) {
        if (uf_ops_.empty()) return formula;
        std::unordered_map<uint64_t, Term> memo;
        return uf_rewrite_rec(formula, memo);
    }

    // Asserts
    void assert_(Term formula)     { solver->assertFormula(uf_rewrite(formula)); }
    void assert_eq(Term t, Term v) { solver->assertFormula(uf_rewrite(equal(t, v))); }

    // Serialize the current assertion stack to a self-contained SMT-LIB2 string.
    // Free constants are collected by DFS (Kind::CONSTANT); literals (FP consts,
    // rounding modes) are inlined by Term::toString() and need no declaration.
    std::string serialize_smt2() { return serialize_smt2(solver->getAssertions()); }
    std::string serialize_smt2(const std::vector<Term>& assertions) {
        std::map<uint64_t, Term> consts;          // dedup by id, sorted for stable output
        std::unordered_set<uint64_t> visited;
        std::vector<Term> stack(assertions.begin(), assertions.end());
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!visited.insert(t.getId()).second) continue;
            if (t.getKind() == Kind::CONSTANT) consts.emplace(t.getId(), t);
            for (const Term& c : t) stack.push_back(c);
        }
        std::string s = "(set-logic ALL)\n";
        for (const auto& kv : consts)
            s += "(declare-fun " + kv.second.toString() + " () "
               + kv.second.getSort().toString() + ")\n";
        for (const Term& a : assertions)
            s += "(assert " + a.toString() + ")\n";
        s += "(check-sat)\n";
        return s;
    }

    // Benchmark hook: gated by SALT_DUMP_SMT2=<dir>, writes each query to
    // <dir>/q<N>.smt2 so cvc5 and bitwuzla can be timed on identical formulas.
    void dump_smt2_if_requested() {
        const char* dir = std::getenv("SALT_DUMP_SMT2");
        if (!dir || !*dir) return;
        static int counter = 0;
        std::string path = std::string(dir) + "/q" + std::to_string(counter++) + ".smt2";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::string s = serialize_smt2();
            std::fwrite(s.data(), 1, s.size(), f);
            std::fclose(f);
        }
    }

    // A product of two non-constant operands, which bvmul in SALT_UF_FIRST abstracts.
    bool has_bvmul_cut(const std::vector<Term>& as) const {
        if (uf_first_ops_.find("bvmul") == std::string::npos) return false;
        std::vector<Term> stack(as.begin(), as.end());
        std::unordered_set<uint64_t> seen;
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!seen.insert(t.getId()).second) continue;
            if (t.getKind() == Kind::BITVECTOR_MULT && t.getNumChildren() == 2 &&
                t[0].getKind() != Kind::CONST_BITVECTOR && t[1].getKind() != Kind::CONST_BITVECTOR)
                return true;
            for (const Term& c : t) stack.push_back(c);
        }
        return false;
    }
    // True if any subterm of the assertion stack has a FloatingPoint sort.  Used
    // by the auto-router: pure-BV queries go to bitwuzla, FP queries to cvc5.
    bool query_has_fp() { return terms_have_fp(solver->getAssertions()); }
    static bool terms_have_fp(const std::vector<Term>& as) {
        std::vector<Term> stack(as.begin(), as.end());
        std::unordered_set<uint64_t> seen;
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!seen.insert(t.getId()).second) continue;
            if (t.getSort().isFloatingPoint()) return true;
            for (const Term& c : t) stack.push_back(c);
        }
        return false;
    }

    // Solve the current query with an external solver binary (bitwuzla) on the
    // serialized SMT2.  Returns Unsat/Sat/Unknown; never populates a cvc5 model.
    CheckResult solve_external(const char* bin, uint64_t ms_override = 0, const std::vector<Term>* as = nullptr) {
        std::string smt2 = as ? serialize_smt2(*as) : serialize_smt2();
        const char* td = std::getenv("TMPDIR");
        std::string dir = (td && *td) ? td : "/tmp";
        std::string tmpl = dir + "/salt_route_XXXXXX";
        std::vector<char> path(tmpl.begin(), tmpl.end()); path.push_back('\0');
        int fd = mkstemp(path.data());
        if (fd < 0) return CheckResult::Unknown;
        if (FILE* tf = fdopen(fd, "w")) {
            std::fwrite(smt2.data(), 1, smt2.size(), tf);
            std::fclose(tf);
        } else { close(fd); unlink(path.data()); return CheckResult::Unknown; }
        std::string cmd = bin;
        if (ms_override > 0) cmd += " -t " + std::to_string(ms_override);
        else if (query_timeout_ms_ > 0) cmd += " -t " + std::to_string(query_timeout_ms_);
        cmd += " "; cmd += path.data(); cmd += " 2>/dev/null";
        CheckResult res = CheckResult::Unknown;
        if (FILE* pp = popen(cmd.c_str(), "r")) {
            char buf[64];
            while (std::fgets(buf, sizeof(buf), pp)) {
                std::string line(buf);
                if (line.rfind("unsat", 0) == 0)   { res = CheckResult::Unsat;   break; }
                if (line.rfind("sat", 0) == 0)      { res = CheckResult::Sat;     break; }
                if (line.rfind("unknown", 0) == 0)  { res = CheckResult::Unknown; break; }
            }
            pclose(pp);
        }
        unlink(path.data());
        return res;
    }

    // In-process cvc5 solve — the source of truth for SAT models (value_str /
    // getValue require it).  The auto-router only ever returns Sat via this path.
    CheckResult check_cvc5() {
        Result r = solver->checkSat();
        print_stats();   // SALT_STATS=1: dump in-process counters (gated inside; survives timeout)
        if (r.isUnsat()) return CheckResult::Unsat;
        if (r.isSat())   return CheckResult::Sat;
        return CheckResult::Unknown;
    }

    // Solving.  SALT_BACKEND=auto routes by query class (pure-BV→bitwuzla via
    // SALT_BITWUZLA, FP→cvc5) with on-Unknown fallback to the other solver.
    // INVARIANT: a Sat return always leaves the cvc5 solver holding a model, so
    // bitwuzla may only short-circuit the Unsat (verified) path.
    // ---- interval axioms for UF-first -------------------------------------
    // An abstracted op's result is unconstrained, so it could be NaN, and every
    // NaN-handling difference (NEON vmaxq propagates NaN, RVV vfmax drops it)
    // resurfaces. When the assumptions bound the operands (finite input bands,
    // constants), the real result lies in a computable interval; asserting that
    // the UF application lies in it is a consequence of the assumptions, hence
    // sound. Intervals are evaluated on the original FP terms, rounding-widened.
    // An interval is a set of float32 values [lo, hi] that excludes NaN; its
    // bounds may be infinite (XNNPACK's "no clamp" bounds are +-inf).
    struct Iv { double lo, hi; };
    // A float32 constant: an FP literal, or to_fp of a 32-bit BV constant (how
    // mk_fp32_from_float builds constants).
    static bool fp32_const(const Term& t, double& v) {
        Term bits;
        if (t.getKind() == Kind::CONST_FLOATINGPOINT) {
            auto [e, sg, b] = t.getFloatingPointValue();
            if (e != 8 || sg != 24) return false;
            bits = b;
        } else if (t.getKind() == Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV &&
                   t[0].getKind() == Kind::CONST_BITVECTOR && t[0].getSort().getBitVectorSize() == 32 &&
                   t.getSort().getFloatingPointExponentSize() == 8) {
            bits = t[0];
        } else {
            return false;
        }
        uint32_t u = (uint32_t)std::stoul(bits.getBitVectorValue(10));
        float f;
        std::memcpy(&f, &u, 4);
        v = f;
        return !std::isnan(v);
    }
    static Iv iv_widen(double lo, double hi) {          // > float32 rounding error
        const double rel = 1.0 / (1 << 21), tiny = 1e-44;
        return {lo - std::fabs(lo) * rel - tiny, hi + std::fabs(hi) * rel + tiny};
    }
    static bool iv_finite(const Iv& i) {
        const double m = std::numeric_limits<float>::max();
        return i.lo >= -m && i.hi <= m && i.lo <= i.hi;
    }
    // Arithmetic on finite operands is NaN-free even when it overflows; bounds past
    // the float32 range become infinite.
    static Iv iv_round(double lo, double hi) {
        const double m = std::numeric_limits<float>::max(), inf = std::numeric_limits<double>::infinity();
        Iv w = iv_widen(lo, hi);
        return {w.lo < -m ? -inf : w.lo, w.hi > m ? inf : w.hi};
    }
    static void iv_meet(std::unordered_map<uint64_t, Iv>& b, const Term& t, double lo, double hi) {
        auto [it, fresh] = b.emplace(t.getId(), Iv{lo, hi});
        if (!fresh) { it->second.lo = std::max(it->second.lo, lo); it->second.hi = std::min(it->second.hi, hi); }
    }
    // Bounds stated by top-level conjunctions of `x <= c`, `x >= c` (c constant).
    // A negated fact: NOT isNaN(t) makes t NaN-free; NOT (x | y) is NOT x and NOT y.
    static void iv_seed_neg(const Term& f, std::unordered_map<uint64_t, Iv>& b,
                            std::vector<std::pair<Term, Term>>& eqs) {
        const double inf = std::numeric_limits<double>::infinity();
        if (f.getKind() == Kind::FLOATINGPOINT_IS_NAN) iv_meet(b, f[0], -inf, inf);
        else if (f.getKind() == Kind::OR) { for (const Term& c : f) iv_seed_neg(c, b, eqs); }
        else if (f.getKind() == Kind::NOT) iv_seed(f[0], b, eqs);
    }
    static void iv_seed(const Term& f, std::unordered_map<uint64_t, Iv>& b,
                        std::vector<std::pair<Term, Term>>& eqs) {
        Kind k = f.getKind();
        const double inf = std::numeric_limits<double>::infinity();
        if (k == Kind::AND) { for (const Term& c : f) iv_seed(c, b, eqs); return; }
        if (k == Kind::NOT) { iv_seed_neg(f[0], b, eqs); return; }
        if (k == Kind::EQUAL && f.getNumChildren() == 2 && f[0].getSort().isFloatingPoint()) {
            eqs.emplace_back(f[0], f[1]);
            return;
        }
        bool le = k == Kind::FLOATINGPOINT_LEQ || k == Kind::FLOATINGPOINT_LT;
        bool ge = k == Kind::FLOATINGPOINT_GEQ || k == Kind::FLOATINGPOINT_GT;
        if ((!le && !ge) || f.getNumChildren() != 2) return;
        double v;
        if (fp32_const(f[1], v)) iv_meet(b, f[0], le ? -inf : v, le ? v : inf);
        else if (fp32_const(f[0], v)) iv_meet(b, f[1], le ? v : -inf, le ? inf : v);
        else { iv_meet(b, f[0], -inf, inf); iv_meet(b, f[1], -inf, inf); }   // a true comparison: no NaN
    }
    // Truth value of a NaN/Inf guard the intervals decide, if any.
    std::optional<bool> iv_cond(const Term& c, const std::unordered_map<uint64_t, Iv>& seeds,
                                std::unordered_map<uint64_t, std::optional<Iv>>& memo) {
        Kind k = c.getKind();
        Iv i;
        if (k == Kind::FLOATINGPOINT_IS_NAN)
            return iv_of(c[0], seeds, memo, i) ? std::optional<bool>(false) : std::nullopt;
        if (k == Kind::FLOATINGPOINT_IS_INF)
            return iv_of(c[0], seeds, memo, i) && iv_finite(i) ? std::optional<bool>(false) : std::nullopt;
        if (k == Kind::NOT) {
            auto v = iv_cond(c[0], seeds, memo);
            return v ? std::optional<bool>(!*v) : std::nullopt;
        }
        if (k == Kind::OR || k == Kind::AND) {
            bool all = true;
            for (const Term& x : c) {
                auto v = iv_cond(x, seeds, memo);
                if (v && *v == (k == Kind::OR)) return k == Kind::OR;   // decided by one child
                all = all && v.has_value();
            }
            return all ? std::optional<bool>(k == Kind::AND) : std::nullopt;
        }
        return std::nullopt;
    }

    bool iv_of(const Term& t, const std::unordered_map<uint64_t, Iv>& seeds,
               std::unordered_map<uint64_t, std::optional<Iv>>& memo, Iv& out) {
        auto m = memo.find(t.getId());
        if (m != memo.end()) { if (m->second) out = *m->second; return m->second.has_value(); }
        memo[t.getId()] = std::nullopt;                 // cycle guard
        std::optional<Iv> r;
        double v;
        Iv a, b, c;
        auto kid = [&](size_t i, Iv& x) { return iv_of(t[i], seeds, memo, x); };
        switch (t.getKind()) {
        case Kind::CONST_FLOATINGPOINT: if (fp32_const(t, v)) r = Iv{v, v}; break;
        // Arithmetic needs finite operands: inf - inf and 0 * inf are NaN.
        case Kind::FLOATINGPOINT_ADD:
            if (kid(1, a) && kid(2, b) && iv_finite(a) && iv_finite(b)) r = iv_round(a.lo + b.lo, a.hi + b.hi);
            break;
        case Kind::FLOATINGPOINT_SUB:
            if (kid(1, a) && kid(2, b) && iv_finite(a) && iv_finite(b)) r = iv_round(a.lo - b.hi, a.hi - b.lo);
            break;
        case Kind::FLOATINGPOINT_MULT:
        case Kind::FLOATINGPOINT_FMA:
            if (kid(1, a) && kid(2, b) && iv_finite(a) && iv_finite(b)) {
                double p[] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
                double lo = *std::min_element(p, p + 4), hi = *std::max_element(p, p + 4);
                if (t.getKind() == Kind::FLOATINGPOINT_MULT) r = iv_round(lo, hi);
                else if (kid(3, c) && iv_finite(c)) r = iv_round(lo + c.lo, hi + c.hi);
            }
            break;
        case Kind::FLOATINGPOINT_NEG: if (kid(0, a)) r = Iv{-a.hi, -a.lo}; break;
        // An integer converted to float lies in the integer's range (rounding-widened): it bounds what a kernel
        // then multiplies by a scale, which matters when two kernels convert back at different widths.
        case Kind::FLOATINGPOINT_TO_FP_FROM_SBV:
        case Kind::FLOATINGPOINT_TO_FP_FROM_UBV:
            if (t.getNumChildren() == 2 && t[1].getSort().isBitVector() && t[1].getSort().getBitVectorSize() <= 32) {
                auto rg = bv_range(t[1]);
                const BvIv& x = t.getKind() == Kind::FLOATINGPOINT_TO_FP_FROM_SBV ? rg.first : rg.second;
                r = iv_round((double)x.lo, (double)x.hi);
            }
            break;
        case Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV:
            // A lane merged at the bit level (vmerge): to_fp(ite(c, x, y)) is
            // to_fp(x) or to_fp(y).
            if (t[0].getKind() == Kind::ITE) {
                Term fx = tm.mkTerm(t.getOp(), {t[0][1]}), fy = tm.mkTerm(t.getOp(), {t[0][2]});
                auto c = iv_cond(t[0][0], seeds, memo);
                if (c) { if (iv_of(*c ? fx : fy, seeds, memo, a)) r = a; }
                else if (iv_of(fx, seeds, memo, a) && iv_of(fy, seeds, memo, b))
                    r = Iv{std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
            }
            break;
        case Kind::FLOATINGPOINT_ABS:
            if (kid(0, a)) r = Iv{(a.lo <= 0 && a.hi >= 0) ? 0.0 : std::min(std::fabs(a.lo), std::fabs(a.hi)),
                                  std::max(std::fabs(a.lo), std::fabs(a.hi))};
            break;
        case Kind::FLOATINGPOINT_MAX:
            if (kid(0, a) && kid(1, b)) r = Iv{std::max(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
        case Kind::FLOATINGPOINT_MIN:
            if (kid(0, a) && kid(1, b)) r = Iv{std::min(a.lo, b.lo), std::min(a.hi, b.hi)}; break;
        case Kind::ITE:
            if (!t[1].getSort().isFloatingPoint()) break;
            // A condition the intervals decide (NEON's NaN guard on finite
            // operands) selects one branch; otherwise take both.
            if (auto c = iv_cond(t[0], seeds, memo)) {
                if (kid(*c ? 1 : 2, a)) r = a;
            } else if (kid(1, a) && kid(2, b)) {
                r = Iv{std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
            }
            break;
        default: break;
        }
        auto sd = seeds.find(t.getId());
        if (sd != seeds.end()) {
            if (r) r = Iv{std::max(r->lo, sd->second.lo), std::min(r->hi, sd->second.hi)};
            else r = sd->second;
        }
        if (r && (!(r->lo <= r->hi) || !t.getSort().isFloatingPoint() ||
                  t.getSort().getFloatingPointExponentSize() != 8 ||
                  t.getSort().getFloatingPointSignificandSize() != 24)) r.reset();
        memo[t.getId()] = r;
        if (r) out = *r;
        return r.has_value();
    }
    Term fp32_bound(double v, bool up) {               // v rounded outward to a float32 constant
        if (std::isinf(v)) {
            uint32_t u = v > 0 ? 0x7f800000u : 0xff800000u;
            return tm.mkFloatingPoint(8, 24, tm.mkBitVector(32, u));
        }
        float f = (float)v;
        if (up && (double)f < v) f = std::nextafter(f, std::numeric_limits<float>::infinity());
        if (!up && (double)f > v) f = std::nextafter(f, -std::numeric_limits<float>::infinity());
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return tm.mkFloatingPoint(8, 24, tm.mkBitVector(32, u));
    }

    // Intervals the assumptions give FP terms: seeds from constant comparisons,
    // pushed through FP equalities (store_fp32_as_bv's to_fp(bv) == value) to a
    // fixed point, then evaluated over the operation graph.
    void interval_state(const std::vector<Term>& as, std::unordered_map<uint64_t, Iv>& seeds,
                        std::vector<std::pair<Term, Term>>& eqs,
                        std::unordered_map<uint64_t, std::optional<Iv>>& ivs) {
        for (const Term& a : as) iv_seed(a, seeds, eqs);
        for (int pass = 0; pass < 4; pass++) {
            ivs.clear();
            size_t before = seeds.size();
            for (auto& [x, y] : eqs) {
                Iv i;
                if (iv_of(y, seeds, ivs, i)) iv_meet(seeds, x, i.lo, i.hi);
                if (iv_of(x, seeds, ivs, i)) iv_meet(seeds, y, i.lo, i.hi);
            }
            if (seeds.size() == before) break;
        }
        nan_free_down(as, seeds, eqs);
        ivs.clear();                                   // final evaluation with every seed
        for (auto& [x, y] : eqs) { Iv i; iv_of(x, seeds, ivs, i); iv_of(y, seeds, ivs, i); }
    }
    // NEON's max/min are ite(isNaN(a) | isNaN(b), NaN, op(a, b)): when such a
    // guard's value is NaN-free, the condition is false, so a, b and op(a, b) are
    // NaN-free too. Pushed to a fixed point through the guards and the FP
    // equalities (store_fp32_as_bv), this takes "this output is not NaN" down to
    // the inputs it reads.
    static void nan_free_down(const std::vector<Term>& as, std::unordered_map<uint64_t, Iv>& seeds,
                              const std::vector<std::pair<Term, Term>>& eqs) {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<Term> guards, stack(as.begin(), as.end());
        std::unordered_set<uint64_t> seen;
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!seen.insert(t.getId()).second) continue;
            if (t.getKind() == Kind::ITE && t[1].getSort().isFloatingPoint() && t[1].isFloatingPointNaN())
                guards.push_back(t);
            for (const Term& c : t) stack.push_back(c);
        }
        std::function<void(const Term&, std::vector<Term>&)> nan_tests = [&](const Term& c, std::vector<Term>& out) {
            if (c.getKind() == Kind::FLOATINGPOINT_IS_NAN) out.push_back(c[0]);
            else if (c.getKind() == Kind::OR) for (const Term& d : c) nan_tests(d, out);
        };
        std::unordered_set<uint64_t> done;
        for (bool changed = true; changed;) {
            changed = false;
            for (const Term& g : guards) {
                if (!seeds.count(g.getId()) || !done.insert(g.getId()).second) continue;
                std::vector<Term> zs{g[2]};
                nan_tests(g[0], zs);
                for (const Term& z : zs) {
                    size_t n = seeds.size();
                    iv_meet(seeds, z, -inf, inf);
                    changed |= seeds.size() != n;
                }
            }
            for (auto& [x, y] : eqs) {
                for (int side = 0; side < 2; side++) {
                    const Term& a = side ? y : x;
                    const Term& b = side ? x : y;
                    auto it = seeds.find(a.getId());
                    if (it == seeds.end() || seeds.count(b.getId())) continue;
                    Iv v = it->second;
                    iv_meet(seeds, b, v.lo, v.hi);
                    changed = true;
                }
            }
        }
    }

    // ---- integer sums ------------------------------------------------------
    // Bit-vector addition is associative and commutative mod 2^n, so a sum is its
    // multiset of addends (sorted by term id). And sext(a + b) = sext(a) + sext(b)
    // (zext alike) whenever a + b cannot overflow, which intervals over the
    // operand widths show for widening accumulations (252 int8 rows in an int16
    // lane). Both rewrites hold in every model: rdsum's 240-row and 252-row
    // groupings become one sum.
    struct BvIv { __int128 lo, hi; };
    std::unordered_map<uint64_t, std::pair<BvIv, BvIv>> bv_iv_memo_;   // signed, unsigned ranges
    static __int128 bv_const(const Term& t) {
        __int128 v = 0;
        for (char c : t.getBitVectorValue(10)) v = v * 10 + (c - '0');
        return v;
    }
    std::pair<BvIv, BvIv> bv_range(const Term& t) {
        auto it = bv_iv_memo_.find(t.getId());
        if (it != bv_iv_memo_.end()) return it->second;
        uint32_t w = t.getSort().getBitVectorSize();
        const __int128 one = 1;
        BvIv S{-(one << (w - 1)), (one << (w - 1)) - 1}, U{0, (one << w) - 1};   // the full ranges
        if (w <= 64) {
            switch (t.getKind()) {
            case Kind::CONST_BITVECTOR: {
                __int128 v = bv_const(t);
                U = {v, v};
                S = v > S.hi ? BvIv{v - (one << w), v - (one << w)} : BvIv{v, v};
                break;
            }
            case Kind::BITVECTOR_SIGN_EXTEND: {
                BvIv s = bv_range(t[0]).first;
                S = s;
                if (s.lo >= 0) U = s;
                break;
            }
            case Kind::BITVECTOR_ZERO_EXTEND:
                if (t.getOp()[0].getUInt32Value() == 0) std::tie(S, U) = bv_range(t[0]);   // the identity
                else U = S = bv_range(t[0]).second;      // at least one leading zero: non-negative
                break;
            case Kind::BITVECTOR_ADD: {
                BvIv s{0, 0}, u{0, 0};
                for (const Term& c : t) {
                    auto [cs, cu] = bv_range(c);
                    s.lo += cs.lo; s.hi += cs.hi; u.lo += cu.lo; u.hi += cu.hi;
                }
                if (s.lo >= S.lo && s.hi <= S.hi) S = s;
                if (u.hi <= U.hi) U = u;
                break;
            }
            case Kind::ITE:
                if (t[1].getSort().isBitVector()) {
                    auto [as, au] = bv_range(t[1]);
                    auto [bs, bu] = bv_range(t[2]);
                    S = {std::min(as.lo, bs.lo), std::max(as.hi, bs.hi)};
                    U = {std::min(au.lo, bu.lo), std::max(au.hi, bu.hi)};
                }
                break;
            case Kind::BITVECTOR_NEG: {                  // -x, unless x can be the minimum (which wraps)
                BvIv s = bv_range(t[0]).first;
                if (s.lo > S.lo) S = {-s.hi, -s.lo};
                break;
            }
            default: break;
            }
        }
        return bv_iv_memo_[t.getId()] = {S, U};
    }
    // A sum's addends cannot overflow its width (signed or unsigned).
    bool add_fits(const Term& t, bool sgn) {
        uint32_t w = t.getSort().getBitVectorSize();
        if (w > 64) return false;
        const __int128 one = 1;
        __int128 lo = 0, hi = 0;
        for (const Term& c : t) {
            auto r = bv_range(c);
            const BvIv& x = sgn ? r.first : r.second;
            lo += x.lo; hi += x.hi;
        }
        return sgn ? (lo >= -(one << (w - 1)) && hi <= (one << (w - 1)) - 1) : (lo >= 0 && hi <= (one << w) - 1);
    }
    // Signed and unsigned ranges of a bit-vector term under the assumptions'
    // bounds on its subterms (params, inputs): bv_seeds reads x OP c atoms, and
    // bv_range_in pushes them through the arithmetic. Each range holds in every
    // model of the assumptions; one that could wrap stays the full range.
    using BvR = std::pair<BvIv, BvIv>;
    static BvR bv_full(uint32_t w) {
        const __int128 one = 1;
        return {{-(one << (w - 1)), (one << (w - 1)) - 1}, {0, (one << w) - 1}};
    }
    std::unordered_map<uint64_t, BvR> bv_seeds(const std::vector<Term>& as) {
        std::unordered_map<uint64_t, BvR> seeds;
        std::unordered_map<uint64_t, BvR> cmemo;
        std::vector<Term> seeded;                          // the bounded terms, in order
        std::vector<Term> atoms(as.begin(), as.end());
        for (size_t i = 0; i < atoms.size(); i++) {
            Term a = atoms[i];
            if (a.getKind() == Kind::AND) { for (const Term& c : a) atoms.push_back(c); continue; }
            bool neg = a.getKind() == Kind::NOT;               // the rewriter writes x >= c as not(x < c)
            if (neg) a = a[0];
            if (a.getNumChildren() != 2 || !a[0].getSort().isBitVector()) continue;
            bool c0 = a[0].getKind() == Kind::CONST_BITVECTOR, c1 = a[1].getKind() == Kind::CONST_BITVECTOR;
            if (c0 == c1 || a[0].getSort().getBitVectorSize() > 64) continue;
            Kind k = a.getKind();
            if (neg) {
                static const std::map<Kind, Kind> negate{
                    {Kind::BITVECTOR_SLT, Kind::BITVECTOR_SGE}, {Kind::BITVECTOR_SGE, Kind::BITVECTOR_SLT},
                    {Kind::BITVECTOR_SLE, Kind::BITVECTOR_SGT}, {Kind::BITVECTOR_SGT, Kind::BITVECTOR_SLE},
                    {Kind::BITVECTOR_ULT, Kind::BITVECTOR_UGE}, {Kind::BITVECTOR_UGE, Kind::BITVECTOR_ULT},
                    {Kind::BITVECTOR_ULE, Kind::BITVECTOR_UGT}, {Kind::BITVECTOR_UGT, Kind::BITVECTOR_ULE}};
                auto f = negate.find(k);
                if (f == negate.end()) continue;
                k = f->second;
            }
            if (c0) {                                       // c OP x  ->  x OP' c
                static const std::map<Kind, Kind> flip{
                    {Kind::BITVECTOR_SGE, Kind::BITVECTOR_SLE}, {Kind::BITVECTOR_SLE, Kind::BITVECTOR_SGE},
                    {Kind::BITVECTOR_SGT, Kind::BITVECTOR_SLT}, {Kind::BITVECTOR_SLT, Kind::BITVECTOR_SGT},
                    {Kind::BITVECTOR_UGE, Kind::BITVECTOR_ULE}, {Kind::BITVECTOR_ULE, Kind::BITVECTOR_UGE},
                    {Kind::BITVECTOR_UGT, Kind::BITVECTOR_ULT}, {Kind::BITVECTOR_ULT, Kind::BITVECTOR_UGT},
                    {Kind::EQUAL, Kind::EQUAL}};
                auto f = flip.find(k);
                if (f == flip.end()) continue;
                k = f->second;
            }
            const Term& x = c1 ? a[0] : a[1];
            auto [cs, cu] = bv_range_in(c1 ? a[1] : a[0], {}, cmemo);
            if (!seeds.count(x.getId())) seeded.push_back(x);
            BvR& r = seeds.try_emplace(x.getId(), bv_full(x.getSort().getBitVectorSize())).first->second;
            BvIv &S = r.first, &U = r.second;
            switch (k) {
            case Kind::BITVECTOR_SGE: S.lo = std::max(S.lo, cs.lo); break;
            case Kind::BITVECTOR_SGT: S.lo = std::max(S.lo, cs.lo + 1); break;
            case Kind::BITVECTOR_SLE: S.hi = std::min(S.hi, cs.hi); break;
            case Kind::BITVECTOR_SLT: S.hi = std::min(S.hi, cs.hi - 1); break;
            case Kind::BITVECTOR_UGE: U.lo = std::max(U.lo, cu.lo); break;
            case Kind::BITVECTOR_UGT: U.lo = std::max(U.lo, cu.lo + 1); break;
            case Kind::BITVECTOR_ULE: U.hi = std::min(U.hi, cu.hi); break;
            case Kind::BITVECTOR_ULT: U.hi = std::min(U.hi, cu.hi - 1); break;
            case Kind::EQUAL: S = cs; U = cu; break;
            default: break;
            }
        }
        // A bounded value's low bits. When x's signed range fits k bits, x is the sign extension of its low
        // k bits, so those bits read as a signed k-bit number equal x and have the same range; when its
        // unsigned range fits k bits, the bits above are zero and the same holds unsigned. A kernel often
        // reads a narrower view of a bounded parameter (the int16 of an int32 field): concat(b1, b0) of a
        // bounded concat(b3, b2, b1, b0), or extract(15, 0) of a bounded 32-bit term.
        const __int128 one = 1;
        for (const Term& x : seeded) {
            const BvR r = seeds.at(x.getId());
            uint32_t w = x.getSort().getBitVectorSize();
            std::vector<Term> lows;
            if (x.getKind() == Kind::BITVECTOR_CONCAT) {
                size_t n = x.getNumChildren();
                for (size_t m = 1; m < n; m++) {             // the last m children
                    std::vector<Term> ch;
                    for (size_t i = n - m; i < n; i++) ch.push_back(x[i]);
                    lows.push_back(m == 1 ? ch[0] : tm.mkTerm(Kind::BITVECTOR_CONCAT, ch));
                }
            }
            for (uint32_t k : {8u, 16u, 32u})
                if (k < w) lows.push_back(tm.mkTerm(tm.mkOp(Kind::BITVECTOR_EXTRACT, {k - 1, 0}), {x}));
            for (const Term& low : lows) {
                uint32_t k = low.getSort().getBitVectorSize();
                bool fs = r.first.lo >= -(one << (k - 1)) && r.first.hi <= (one << (k - 1)) - 1;
                bool fu = r.second.lo >= 0 && r.second.hi <= (one << k) - 1;
                if (!fs && !fu) continue;
                BvR& lr = seeds.try_emplace(low.getId(), bv_full(k)).first->second;
                if (fs) lr.first = {std::max(lr.first.lo, r.first.lo), std::min(lr.first.hi, r.first.hi)};
                if (fu) lr.second = {std::max(lr.second.lo, r.second.lo), std::min(lr.second.hi, r.second.hi)};
            }
        }
        return seeds;
    }
    BvR bv_range_in(const Term& t, const std::unordered_map<uint64_t, BvR>& seeds,
                    std::unordered_map<uint64_t, BvR>& memo) {
        auto it = memo.find(t.getId());
        if (it != memo.end()) return it->second;
        uint32_t w = t.getSort().getBitVectorSize();
        if (w > 64) return {{0, -1}, {0, -1}};             // no range kept (callers check widths first)
        auto [S, U] = bv_full(w);
        const BvR full = bv_full(w);
        auto fits_s = [&](const BvIv& r) { return r.lo >= full.first.lo && r.hi <= full.first.hi; };
        auto fits_u = [&](const BvIv& r) { return r.lo >= 0 && r.hi <= full.second.hi; };
        auto rng = [&](const Term& c) { return bv_range_in(c, seeds, memo); };
        auto amount = [&](const Term& c, uint32_t& v) {
            if (c.getKind() != Kind::CONST_BITVECTOR) return false;
            __int128 x = bv_const(c);
            v = x >= w ? w : (uint32_t)x;
            return true;
        };
        // Products and shifts stay inside __int128 while each factor is at most 2^63.
        const __int128 lim = (__int128)1 << 63;
        auto small = [&](const BvIv& a) { return a.lo >= -lim && a.hi <= lim; };
        auto mul = [&](const BvIv& a, const BvIv& b, bool& ok) {
            if (!small(a) || !small(b)) { ok = false; return BvIv{0, 0}; }
            __int128 c[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
            return BvIv{*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
        };
        const __int128 one = 1;
        uint32_t v = 0;
        {
            switch (t.getKind()) {
            case Kind::CONST_BITVECTOR: {
                __int128 x = bv_const(t);
                U = {x, x};
                S = x > full.first.hi ? BvIv{x - (one << w), x - (one << w)} : BvIv{x, x};
                break;
            }
            case Kind::BITVECTOR_SIGN_EXTEND: {
                BvIv s = rng(t[0]).first;
                S = s;
                if (s.lo >= 0) U = s;
                break;
            }
            case Kind::BITVECTOR_ZERO_EXTEND:
                if (t.getOp()[0].getUInt32Value() == 0) std::tie(S, U) = rng(t[0]);
                else U = S = rng(t[0]).second;
                break;
            case Kind::BITVECTOR_EXTRACT:
                if (t.getOp()[1].getUInt32Value() == 0 && t[0].getSort().getBitVectorSize() <= 64) {
                    // the low bits keep a value that fits them
                    auto [cs, cu] = rng(t[0]);
                    if (fits_s(cs)) { S = cs; if (cs.lo >= 0) U = cs; }
                    if (fits_u(cu)) { U = cu; if (cu.hi <= full.first.hi) S = cu; }
                }
                break;
            case Kind::BITVECTOR_CONCAT: {                   // signed(x) * 2^k + unsigned(y)
                std::tie(S, U) = rng(t[0]);
                for (size_t i = 1; i < t.getNumChildren(); i++) {
                    uint32_t k = t[i].getSort().getBitVectorSize();
                    BvIv cu = rng(t[i]).second;
                    S = {S.lo * (one << k) + cu.lo, S.hi * (one << k) + cu.hi};
                    U = {U.lo * (one << k) + cu.lo, U.hi * (one << k) + cu.hi};
                }
                break;
            }
            case Kind::BITVECTOR_ADD: {
                BvIv s{0, 0}, u{0, 0};
                for (const Term& c : t) {
                    auto [cs, cu] = rng(c);
                    s = {s.lo + cs.lo, s.hi + cs.hi};
                    u = {u.lo + cu.lo, u.hi + cu.hi};
                }
                if (fits_s(s)) S = s;
                if (fits_u(u)) U = u;
                break;
            }
            case Kind::BITVECTOR_SUB:
                if (t.getNumChildren() == 2) {
                    auto [as, au] = rng(t[0]);
                    auto [bs, bu] = rng(t[1]);
                    BvIv s{as.lo - bs.hi, as.hi - bs.lo}, u{au.lo - bu.hi, au.hi - bu.lo};
                    if (fits_s(s)) S = s;
                    if (fits_u(u)) U = u;
                }
                break;
            case Kind::BITVECTOR_NEG: {
                BvIv s = rng(t[0]).first, n{-s.hi, -s.lo};
                if (fits_s(n)) S = n;
                break;
            }
            case Kind::BITVECTOR_MULT: {
                BvIv s{1, 1}, u{1, 1};
                bool ok_s = true, ok_u = true;
                for (const Term& c : t) {
                    auto [cs, cu] = rng(c);
                    s = mul(s, cs, ok_s); u = mul(u, cu, ok_u);
                    ok_s = ok_s && fits_s(s); ok_u = ok_u && fits_u(u);
                    if (!ok_s) s = {0, 0};
                    if (!ok_u) u = {0, 0};
                }
                if (ok_s) S = s;
                if (ok_u) U = u;
                break;
            }
            case Kind::BITVECTOR_SHL:
                if (amount(t[1], v)) {
                    if (v >= w) { S = U = {0, 0}; break; }
                    auto [as, au] = rng(t[0]);
                    if (v <= 62 && small(as)) { BvIv s{as.lo * (one << v), as.hi * (one << v)}; if (fits_s(s)) S = s; }
                    if (v <= 62 && small(au)) { BvIv u{au.lo * (one << v), au.hi * (one << v)}; if (fits_u(u)) U = u; }
                }
                break;
            case Kind::BITVECTOR_ASHR:
                if (amount(t[1], v)) {
                    BvIv as = rng(t[0]).first;
                    uint32_t k = std::min(v, w - 1);
                    S = {as.lo >> k, as.hi >> k};
                }
                break;
            case Kind::BITVECTOR_LSHR:
                if (amount(t[1], v)) {
                    if (v >= w) { S = U = {0, 0}; break; }
                    BvIv au = rng(t[0]).second;
                    U = {au.lo >> v, au.hi >> v};
                }
                break;
            case Kind::ITE:
                if (t[1].getSort().isBitVector()) {
                    auto [as, au] = rng(t[1]);
                    auto [bs, bu] = rng(t[2]);
                    S = {std::min(as.lo, bs.lo), std::max(as.hi, bs.hi)};
                    U = {std::min(au.lo, bu.lo), std::max(au.hi, bu.hi)};
                }
                break;
            default: break;
            }
        }
        if (auto sd = seeds.find(t.getId()); sd != seeds.end()) {
            S = {std::max(S.lo, sd->second.first.lo), std::min(S.hi, sd->second.first.hi)};
            U = {std::max(U.lo, sd->second.second.lo), std::min(U.hi, sd->second.second.hi)};
        }
        // A non-negative signed range is the unsigned one, and the reverse.
        if (S.lo >= 0) U = {std::max(U.lo, S.lo), std::min(U.hi, S.hi)};
        if (U.hi <= full.first.hi) S = {std::max(S.lo, U.lo), std::min(S.hi, U.hi)};
        return memo[t.getId()] = {S, U};
    }
    Term flat_add(const std::vector<Term>& ch) {
        std::vector<Term> leaves;
        std::function<void(const Term&)> collect = [&](const Term& c) {
            if (c.getKind() == Kind::BITVECTOR_ADD) { for (const Term& d : c) collect(d); }
            else leaves.push_back(c);
        };
        for (const Term& c : ch) collect(c);
        std::sort(leaves.begin(), leaves.end(), [](const Term& a, const Term& b) { return a.getId() < b.getId(); });
        return leaves.size() == 1 ? leaves[0] : tm.mkTerm(Kind::BITVECTOR_ADD, leaves);
    }
    Term push_ext(bool sgn, uint32_t k, const Term& x) {
        if (k == 0) return x;
        Kind ek = sgn ? Kind::BITVECTOR_SIGN_EXTEND : Kind::BITVECTOR_ZERO_EXTEND;
        if (x.getKind() == ek) return push_ext(sgn, k + x.getOp()[0].getUInt32Value(), x[0]);
        if (x.getKind() == Kind::BITVECTOR_ADD && add_fits(x, sgn)) {
            std::vector<Term> ch;
            for (const Term& c : x) ch.push_back(push_ext(sgn, k, c));
            return flat_add(ch);
        }
        if (ring_mode_ && x.getKind() == Kind::CONST_BITVECTOR && x.getSort().getBitVectorSize() + k <= 64) {
            uint32_t w0 = x.getSort().getBitVectorSize();
            __int128 v = bv_const(x);
            if (sgn && v >= ((__int128)1 << (w0 - 1))) v -= (__int128)1 << w0;
            return ring_const(w0 + k, v);
        }
        // sext(-y) = -sext(y) unless y is the minimum, whose negation wraps.
        if (sgn && ring_mode_ && x.getKind() == Kind::BITVECTOR_NEG) {
            uint32_t w = x.getSort().getBitVectorSize();
            if (w <= 64 && bv_range(x[0]).first.lo > -((__int128)1 << (w - 1))) return ring_neg(push_ext(sgn, k, x[0]));
        }
        return tm.mkTerm(tm.mkOp(ek, {k}), {x});
    }

    // ---- ring normal form -------------------------------------------------
    // Bit-vector arithmetic is arithmetic mod 2^w, a commutative ring: a - b = a + (-b), negation and
    // multiplication distribute over sums, and the low n bits of a sum, negation or product are that operation
    // on the operands' low n bits. Rewriting every term to a flat sum of products of non-sums makes a kernel that
    // computes x*m - zp*m (a folded bias) syntactically equal to one that computes (x - zp)*m, which the bvmul
    // abstraction needs: an uninterpreted product only matches a product of the same operands. Every rewrite is
    // an identity of the ring (the extension rules are range-checked), so the result has the same models.
    bool ring_mode_ = false;
    size_t ring_rewrites_ = 0;
    Term ring_const(uint32_t w, __int128 v) {            // v mod 2^w, w <= 64
        return bv_val(w, (uint64_t)v & (w >= 64 ? ~0ULL : ((1ULL << w) - 1)));
    }
    static bool ring_is_const(const Term& x) {
        return x.getKind() == Kind::CONST_BITVECTOR && x.getSort().getBitVectorSize() <= 64;
    }
    // A flat sum with its constants added up (and a zero dropped).
    Term ring_add(const std::vector<Term>& ch) {
        Term sum = flat_add(ch);
        if (sum.getKind() != Kind::BITVECTOR_ADD) return sum;
        uint32_t w = sum.getSort().getBitVectorSize();
        if (w > 64) return sum;
        __int128 c = 0;
        std::vector<Term> rest;
        size_t consts = 0;
        for (const Term& t : sum) {
            if (ring_is_const(t)) { c += bv_const(t); consts++; }
            else rest.push_back(t);
        }
        if (consts == 0) return sum;
        Term k = ring_const(w, c);
        if (bv_const(k) != 0 || rest.empty()) rest.push_back(k);
        return rest.size() == 1 ? rest[0] : flat_add(rest);
    }
    Term ring_neg(const Term& x) {
        if (x.getKind() == Kind::BITVECTOR_NEG) return x[0];
        if (ring_is_const(x)) return ring_const(x.getSort().getBitVectorSize(), -bv_const(x));
        if (x.getKind() == Kind::BITVECTOR_ADD) {
            std::vector<Term> ch;
            for (const Term& c : x) ch.push_back(ring_neg(c));
            return ring_add(ch);
        }
        return tm.mkTerm(Kind::BITVECTOR_NEG, {x});
    }
    static size_t ring_addends(const Term& x) { return x.getKind() == Kind::BITVECTOR_ADD ? x.getNumChildren() : 1; }
    Term ring_mul(const Term& a, const Term& b) {
        if (a.getKind() == Kind::BITVECTOR_NEG) return ring_neg(ring_mul(a[0], b));
        if (b.getKind() == Kind::BITVECTOR_NEG) return ring_neg(ring_mul(a, b[0]));
        if (ring_is_const(a) || ring_is_const(b)) {      // 0 x = 0, 1 x = x, c d folded
            const Term& k = ring_is_const(a) ? a : b;
            const Term& o = ring_is_const(a) ? b : a;
            uint32_t w = k.getSort().getBitVectorSize();
            __int128 kv = bv_const(k);
            if (kv == 0) return k;
            if (kv == 1) return o;
            if (ring_is_const(o)) {
                unsigned __int128 p = (unsigned __int128)kv * (unsigned __int128)bv_const(o);
                return ring_const(w, (__int128)(p & (((unsigned __int128)1 << 64) - 1)));
            }
        }
        size_t n = ring_addends(a) * ring_addends(b);
        if (n > 1 && n <= 16) {                          // (a1 + a2) b = a1 b + a2 b; bounded, so no blow-up
            std::vector<Term> ch;
            if (a.getKind() == Kind::BITVECTOR_ADD) for (const Term& c : a) ch.push_back(ring_mul(c, b));
            else for (const Term& c : b) ch.push_back(ring_mul(a, c));
            return ring_add(ch);
        }
        // A product of two sign-extended values is taken at the width it cannot overflow (the operands' widths
        // added) and extended from there, so a 16-bit widening multiply of two int8 and a 32-bit multiply of the
        // same two values extended are one term.
        if (a.getKind() == Kind::BITVECTOR_SIGN_EXTEND && b.getKind() == Kind::BITVECTOR_SIGN_EXTEND) {
            uint32_t w = a.getSort().getBitVectorSize();
            uint32_t wa = a[0].getSort().getBitVectorSize(), wb = b[0].getSort().getBitVectorSize();
            if (wa + wb < w) {
                Term x = push_ext(true, wb, a[0]), y = push_ext(true, wa, b[0]);
                Term m = x.getId() <= y.getId() ? tm.mkTerm(Kind::BITVECTOR_MULT, {x, y}) : tm.mkTerm(Kind::BITVECTOR_MULT, {y, x});
                return tm.mkTerm(tm.mkOp(Kind::BITVECTOR_SIGN_EXTEND, {w - wa - wb}), {m});
            }
        }
        return a.getId() <= b.getId() ? tm.mkTerm(Kind::BITVECTOR_MULT, {a, b}) : tm.mkTerm(Kind::BITVECTOR_MULT, {b, a});
    }
    Term ring_low(uint32_t n, const Term& x) {           // the low n bits of x
        uint32_t w = x.getSort().getBitVectorSize();
        if (n == w) return x;
        if (ring_is_const(x)) return ring_const(n, bv_const(x));
        switch (x.getKind()) {
        case Kind::BITVECTOR_ADD: {
            std::vector<Term> ch;
            for (const Term& c : x) ch.push_back(ring_low(n, c));
            return ring_add(ch);
        }
        case Kind::BITVECTOR_NEG: return ring_neg(ring_low(n, x[0]));
        case Kind::BITVECTOR_MULT: {
            Term r = ring_low(n, x[0]);
            for (size_t i = 1; i < x.getNumChildren(); i++) r = ring_mul(r, ring_low(n, x[i]));
            return r;
        }
        case Kind::BITVECTOR_SIGN_EXTEND:
        case Kind::BITVECTOR_ZERO_EXTEND: {
            uint32_t w0 = x[0].getSort().getBitVectorSize();
            if (n <= w0) return ring_low(n, x[0]);
            return push_ext(x.getKind() == Kind::BITVECTOR_SIGN_EXTEND, n - w0, x[0]);
        }
        default: return tm.mkTerm(tm.mkOp(Kind::BITVECTOR_EXTRACT, {n - 1, 0}), {x});
        }
    }
    std::vector<Term> ring_normal(const std::vector<Term>& as0) {
        // A term the assumptions fix to one value (a path condition zero_point == 0) is replaced by that value, so
        // a kernel's specialized path and the general formula meet (the rewrites below fold the constants; cvc5's
        // simplifier is not used, as it would also reshape the terms the assumptions' ranges are keyed by). The
        // atoms that fix it are top-level assertions, so every model agrees and the substitution keeps the models.
        std::vector<Term> as = as0;
        {
            std::unordered_map<uint64_t, BvR> seeds = bv_seeds(as0);
            bool any = false;
            for (const auto& [id, r] : seeds) {
                const BvIv& S = r.first;
                const BvIv& U = r.second;
                if (S.lo != S.hi && U.lo != U.hi) continue;
                Term x;
                for (const Term& a : as0)
                    if (find_by_id(a, id, x)) break;
                if (x.isNull() || !x.getSort().isBitVector() || x.getKind() == Kind::CONST_BITVECTOR) continue;
                uint32_t w = x.getSort().getBitVectorSize();
                if (w > 64) continue;
                __int128 v = S.lo == S.hi ? S.lo : U.lo;
                // The value fixes what the term is built from: the operand of an extension, the parts of a concat.
                std::function<void(const Term&, uint64_t)> fix = [&](const Term& t, uint64_t bits) {
                    uint32_t tw = t.getSort().getBitVectorSize();
                    if (tw < 64) bits &= (1ULL << tw) - 1;
                    if (t.getKind() == Kind::CONST_BITVECTOR) return;
                    if (t.getKind() == Kind::BITVECTOR_SIGN_EXTEND || t.getKind() == Kind::BITVECTOR_ZERO_EXTEND) {
                        fix(t[0], bits);
                    } else if (t.getKind() == Kind::BITVECTOR_CONCAT) {
                        uint32_t lo = 0;
                        for (size_t i = t.getNumChildren(); i-- > 0;) {
                            fix(t[i], lo < 64 ? bits >> lo : 0);
                            lo += t[i].getSort().getBitVectorSize();
                        }
                    } else {
                        Term val = bv_val(tw, bits);
                        for (Term& a : as) a = a.substitute(t, val);
                        any = true;
                    }
                };
                fix(x, (uint64_t)v);
            }
            if (any) ring_rewrites_++;
        }
        std::unordered_map<uint64_t, Term> memo;
        ring_mode_ = true;
        std::function<Term(const Term&)> rec = [&](const Term& t) -> Term {
            if (t.getNumChildren() == 0) return t;
            auto it = memo.find(t.getId());
            if (it != memo.end()) return it->second;
            Kind k = t.getKind();
            std::vector<Term> ch;
            for (const Term& c : t) ch.push_back(rec(c));
            Term plain = t.hasOp() ? tm.mkTerm(t.getOp(), ch) : tm.mkTerm(k, ch);
            Term out = plain;
            switch (k) {
            case Kind::BITVECTOR_SUB: out = ring_add({ch[0], ring_neg(ch[1])}); break;
            case Kind::BITVECTOR_ADD: out = ring_add(ch); break;
            case Kind::BITVECTOR_NEG: out = ring_neg(ch[0]); break;
            case Kind::BITVECTOR_MULT:
                out = ch[0];
                for (size_t i = 1; i < ch.size(); i++) out = ring_mul(out, ch[i]);
                break;
            case Kind::BITVECTOR_SIGN_EXTEND:
            case Kind::BITVECTOR_ZERO_EXTEND:
                out = push_ext(k == Kind::BITVECTOR_SIGN_EXTEND, t.getOp()[0].getUInt32Value(), ch[0]);
                break;
            case Kind::BITVECTOR_EXTRACT:
                if (t.getOp()[1].getUInt32Value() == 0) out = ring_low(t.getOp()[0].getUInt32Value() + 1, ch[0]);
                break;
            default: break;
            }
            if (out != plain) ring_rewrites_++;
            memo[t.getId()] = out;
            return out;
        };
        std::vector<Term> out;
        for (const Term& a : as) out.push_back(rec(a));
        ring_mode_ = false;
        return out;
    }

    // NEON's vbsl with a compare mask, (m & x) | (~m & y) for m = ite(c, ~0, 0), is
    // ite(c, x, y), which is how RVV's vmerge is modeled: an equivalence, so the
    // two kernels' select chains become the same term.
    Term bit_select(const std::vector<Term>& ch, const Term& orig) {
        if (ch.size() != 2) return orig;
        auto full_mask = [](const Term& m) {
            if (m.getKind() != Kind::ITE || m[1].getKind() != Kind::CONST_BITVECTOR ||
                m[2].getKind() != Kind::CONST_BITVECTOR) return false;
            std::string ones = m[1].getBitVectorValue(2), zeros = m[2].getBitVectorValue(2);
            return ones.find('0') == std::string::npos && zeros.find('1') == std::string::npos;
        };
        for (int s = 0; s < 2; s++) {
            const Term &p = ch[s], &q = ch[1 - s];
            if (p.getKind() != Kind::BITVECTOR_AND || q.getKind() != Kind::BITVECTOR_AND ||
                p.getNumChildren() != 2 || q.getNumChildren() != 2) continue;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    const Term &m = p[i], &nm = q[j];
                    if (full_mask(m) && nm.getKind() == Kind::BITVECTOR_NOT && nm[0] == m)
                        return tm.mkTerm(Kind::ITE, {m[0], p[1 - i], q[1 - j]});
                }
        }
        return orig;
    }

    // isNaN / isInf of a term the assumptions bound to a finite interval is false
    // in every model, so rewriting them to false (and letting the rewriter fold
    // the NaN-handling ite chains) gives a query with exactly the same models.
    size_t int_rewrites_ = 0;
    std::vector<Term> interval_simplified(const std::vector<Term>& as, bool int_sums = false) {
        std::unordered_map<uint64_t, Iv> seeds;
        std::vector<std::pair<Term, Term>> eqs;
        std::unordered_map<uint64_t, std::optional<Iv>> ivs;
        interval_state(as, seeds, eqs, ivs);
        std::unordered_map<uint64_t, Term> memo;
        std::function<Term(const Term&)> rec = [&](const Term& t) -> Term {
            if (t.getNumChildren() == 0) return t;
            auto it = memo.find(t.getId());
            if (it != memo.end()) return it->second;
            Kind k = t.getKind();
            Term out;
            Iv i;
            if ((k == Kind::FLOATINGPOINT_IS_NAN || k == Kind::FLOATINGPOINT_IS_INF) &&
                iv_of(t[0], seeds, ivs, i) && (k == Kind::FLOATINGPOINT_IS_NAN || iv_finite(i))) {
                out = tm.mkFalse();
            } else {
                std::vector<Term> ch;
                for (const Term& c : t) ch.push_back(rec(c));
                out = t.hasOp() ? tm.mkTerm(t.getOp(), ch) : tm.mkTerm(k, ch);
                Term plain = out;
                if (k == Kind::BITVECTOR_OR) out = bit_select(ch, out);
                else if (k == Kind::BITVECTOR_ADD && int_sums) out = flat_add(ch);
                else if ((k == Kind::BITVECTOR_SIGN_EXTEND || k == Kind::BITVECTOR_ZERO_EXTEND) && int_sums)
                    out = push_ext(k == Kind::BITVECTOR_SIGN_EXTEND, t.getOp()[0].getUInt32Value(), ch[0]);
                if (out != plain) int_rewrites_++;
            }
            memo[t.getId()] = out;
            return out;
        };
        std::vector<Term> out;
        for (const Term& a : as) out.push_back(solver->simplify(rec(a)));
        if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1') {
            size_t tests = 0, folded = 0;
            std::unordered_set<uint64_t> seen;
            std::vector<Term> st(as.begin(), as.end());
            while (!st.empty()) {
                Term t = st.back(); st.pop_back();
                if (!seen.insert(t.getId()).second) continue;
                if (t.getKind() == Kind::FLOATINGPOINT_IS_NAN || t.getKind() == Kind::FLOATINGPOINT_IS_INF) {
                    tests++;
                    auto m = memo.find(t.getId());
                    bool f = m != memo.end() && m->second.isBooleanValue();
                    folded += f;
                }
                for (const Term& c : t) st.push_back(c);
            }
            std::fprintf(stderr, "[simplify] %zu isNaN/isInf tests, %zu folded to false, %zu seeds\n", tests, folded, seeds.size());
        }
        return out;
    }

    // ---- AC normal form of fp.max / fp.min ----------------------------------
    // fp.max and fp.min (RVV vfmax/vfmin) are each commutative and associative on every input, NaN included: a
    // NaN operand is ignored, two NaNs give NaN. So a tree of one of them is the set of its leaves, and rewritten
    // as one chain over the leaves sorted by term id, a max tree and a max chain over the same values are one term
    // (Phase 1's nine-input max tree and a Phase-2 chain that starts from the clamp bound). Unlike the lattice
    // form below this needs no NaN-free leaves, but it does not distribute max over min. The rewrite keeps the
    // hardware's semantics (vfmax orders -0 below +0, so it is associative there); SMT-LIB leaves
    // fp.max(+0, -0) unspecified, so, as with the lattice form, only an Unsat answer is used.
    std::vector<Term> ac_minmax_normalized(const std::vector<Term>& as, size_t& rewrites) {
        std::unordered_map<uint64_t, Term> memo;
        std::function<Term(const Term&)> rec = [&](const Term& t) -> Term {
            if (t.getNumChildren() == 0) return t;
            auto it = memo.find(t.getId());
            if (it != memo.end()) return it->second;
            Kind k = t.getKind();
            std::vector<Term> ch;
            for (const Term& c : t) ch.push_back(rec(c));
            Term out = t.hasOp() ? tm.mkTerm(t.getOp(), ch) : tm.mkTerm(k, ch);
            if (k == Kind::FLOATINGPOINT_MAX || k == Kind::FLOATINGPOINT_MIN) {
                std::vector<Term> leaves;
                std::function<void(const Term&)> collect = [&](const Term& c) {
                    if (c.getKind() == k) { for (const Term& d : c) collect(d); }
                    else leaves.push_back(c);
                };
                for (const Term& c : ch) collect(c);
                std::sort(leaves.begin(), leaves.end(), [](const Term& a, const Term& b) { return a.getId() < b.getId(); });
                leaves.erase(std::unique(leaves.begin(), leaves.end(),
                                         [](const Term& a, const Term& b) { return a.getId() == b.getId(); }), leaves.end());
                Term acc = leaves[0];
                for (size_t i = 1; i < leaves.size(); i++) acc = tm.mkTerm(k, {acc, leaves[i]});
                if (acc != out) rewrites++;
                out = acc;
            }
            memo[t.getId()] = out;
            return out;
        };
        std::vector<Term> out;
        for (const Term& a : as) out.push_back(rec(a));
        return out;
    }

    // ---- max/min normal form ----------------------------------------------
    // On non-NaN operands, ARM FMAX/FMIN and RISC-V vfmax/vfmin order -0 below +0,
    // so max and min are the join and meet of a total order: associative,
    // commutative, idempotent and distributive. An identity holds in every chain
    // iff it holds in every distributive lattice, iff both sides have the same
    // irredundant join-of-meets form, so rewriting each max/min tree over NaN-free
    // leaves to that form (clauses sorted by term id) maps NEON's max tree and
    // RVV's max chain to one term. SMT-LIB leaves fp.max(+0, -0) unspecified: the
    // hardware's choice is one interpretation of the rewritten query, not of the
    // original's models, so only an Unsat answer is used. `merged` counts distinct
    // max/min trees that became the same term; when it is 0 (a clamp on both
    // sides) the form cannot help and the stage is skipped.
    // `facts` (optional) are further assertions of the same query to take
    // NaN-freeness from: simplification folds a case's "ref is not NaN" away.
    std::vector<Term> lattice_normalized(const std::vector<Term>& as, size_t& merged,
                                         const std::vector<Term>* facts = nullptr) {
        std::unordered_map<uint64_t, Iv> seeds;
        std::vector<std::pair<Term, Term>> eqs;
        std::unordered_map<uint64_t, std::optional<Iv>> ivs;
        if (facts) {
            // NaN-freeness derived on the unsimplified facts, carried to the
            // simplified form of each such term (an equivalent term).
            std::unordered_map<uint64_t, Iv> fs;
            std::vector<std::pair<Term, Term>> feqs;
            std::unordered_map<uint64_t, std::optional<Iv>> fivs;
            interval_state(*facts, fs, feqs, fivs);
            std::unordered_set<uint64_t> seen;
            std::vector<Term> st(facts->begin(), facts->end());
            while (!st.empty()) {
                Term t = st.back(); st.pop_back();
                if (!seen.insert(t.getId()).second) continue;
                auto it = fs.find(t.getId());
                if (it != fs.end()) {
                    iv_meet(seeds, t, it->second.lo, it->second.hi);
                    iv_meet(seeds, solver->simplify(t), it->second.lo, it->second.hi);
                }
                for (const Term& c : t) st.push_back(c);
            }
        }
        interval_state(as, seeds, eqs, ivs);
        using Clause = std::vector<Term>;               // a meet, leaves sorted by id
        using Dnf = std::vector<Clause>;                // a join of meets
        const size_t max_clauses = 256;
        auto ids = [](const Clause& c) { std::vector<uint64_t> v; for (const Term& t : c) v.push_back(t.getId()); return v; };
        auto subset = [&](const Clause& a, const Clause& b) {   // a's leaves all in b
            auto x = ids(a), y = ids(b);
            return std::includes(y.begin(), y.end(), x.begin(), x.end());
        };
        auto irredundant = [&](Dnf d) {
            for (Clause& c : d) {
                std::sort(c.begin(), c.end(), [](const Term& a, const Term& b) { return a.getId() < b.getId(); });
                c.erase(std::unique(c.begin(), c.end(), [](const Term& a, const Term& b) { return a.getId() == b.getId(); }), c.end());
            }
            std::sort(d.begin(), d.end(), [&](const Clause& a, const Clause& b) { return ids(a) < ids(b); });
            d.erase(std::unique(d.begin(), d.end(), [&](const Clause& a, const Clause& b) { return ids(a) == ids(b); }), d.end());
            Dnf out;
            for (size_t i = 0; i < d.size(); i++) {      // absorption: x | (x & y) = x
                bool absorbed = false;
                for (size_t j = 0; j < d.size() && !absorbed; j++)
                    absorbed = j != i && d[j].size() < d[i].size() && subset(d[j], d[i]);
                if (!absorbed) out.push_back(d[i]);
            }
            return out;
        };
        auto lattice_op = [](const Term& t) {
            Kind k = t.getKind();
            return (k == Kind::FLOATINGPOINT_MAX || k == Kind::FLOATINGPOINT_MIN) && t.getSort().isFloatingPoint() &&
                   t.getSort().getFloatingPointExponentSize() == 8 && t.getSort().getFloatingPointSignificandSize() == 24;
        };
        std::unordered_map<uint64_t, Term> memo;
        std::unordered_set<uint64_t> roots, forms;
        std::function<Term(const Term&)> rec;
        // The join-of-meets form of t, or nullopt when a leaf may be NaN (fp.max
        // returns the other operand, which breaks absorption) or the form is too big.
        std::function<std::optional<Dnf>(const Term&)> dnf = [&](const Term& t) -> std::optional<Dnf> {
            if (!lattice_op(t)) {
                Iv i;
                if (!iv_of(t, seeds, ivs, i)) return std::nullopt;
                return Dnf{Clause{rec(t)}};
            }
            auto a = dnf(t[0]);
            if (!a) return std::nullopt;
            auto b = dnf(t[1]);
            if (!b) return std::nullopt;
            Dnf r;
            if (t.getKind() == Kind::FLOATINGPOINT_MAX) {
                r = *a;
                r.insert(r.end(), b->begin(), b->end());
            } else {
                if (a->size() * b->size() > max_clauses) return std::nullopt;
                for (const Clause& x : *a)
                    for (const Clause& y : *b) {
                        Clause c = x;
                        c.insert(c.end(), y.begin(), y.end());
                        r.push_back(c);
                    }
            }
            r = irredundant(r);
            if (r.size() > max_clauses) return std::nullopt;
            return r;
        };
        rec = [&](const Term& t) -> Term {
            if (t.getNumChildren() == 0) return t;
            auto it = memo.find(t.getId());
            if (it != memo.end()) return it->second;
            Term out;
            std::optional<Dnf> d = lattice_op(t) ? dnf(t) : std::nullopt;
            if (d) {
                for (const Clause& c : *d) {
                    Term m = c[0];
                    for (size_t i = 1; i < c.size(); i++) m = tm.mkTerm(Kind::FLOATINGPOINT_MIN, {m, c[i]});
                    out = out.isNull() ? m : tm.mkTerm(Kind::FLOATINGPOINT_MAX, {out, m});
                }
                roots.insert(t.getId());
                forms.insert(out.getId());
            } else {
                std::vector<Term> ch;
                for (const Term& c : t) ch.push_back(rec(c));
                out = t.hasOp() ? tm.mkTerm(t.getOp(), ch) : tm.mkTerm(t.getKind(), ch);
            }
            memo[t.getId()] = out;
            return out;
        };
        std::vector<Term> out;
        for (const Term& a : as) out.push_back(rec(a));
        merged = roots.size() - forms.size();
        if (const char* dbg = std::getenv("SALT_UF_DEBUG"); dbg && *dbg == '1')
            std::fprintf(stderr, "[lattice] %zu max/min trees, %zu normal forms\n", roots.size(), forms.size());
        return out;
    }

    // An over-approximating query (UF-first, lattice normal form) in a fresh
    // solver on the same TermManager: only an Unsat answer is meaningful.
    CheckResult check_unsat_only(const std::vector<Term>& as, uint64_t ms) {
        Solver s(tm);
        s.setOption("fp-exp", "true");
        if (ms) s.setOption("tlimit-per", std::to_string(ms));
        for (const Term& a : as) s.assertFormula(a);
        return s.checkSat().isUnsat() ? CheckResult::Unsat : CheckResult::Unknown;
    }

    // ---- per-output split (SALT_NAN_POLICY=ref) ----------------------------
    // Under XNNPACK's NaN policy the goal is NOT AND_e (isNaN(ref_e) | ref_e = cand_e).
    // Each disjunct of its negation is solved alone: there "ref_e is not NaN" is a
    // top-level fact, which nan_free_down takes to the inputs ref_e reads, so the
    // normal forms that need NaN-free operands apply. Each case keeps only the
    // definitional equalities its goal reaches (dropping the rest only weakens it,
    // and a dropped to_fp(c) = t can always be met by choosing c). Every case
    // UNSAT is the whole query UNSAT.
    static bool nan_ref_policy() {
        static const bool on = [] { const char* s = std::getenv("SALT_NAN_POLICY"); return s && std::string(s) == "ref"; }();
        return on;
    }
    static void conjuncts(const Term& t, std::vector<Term>& out) {
        if (t.getKind() == Kind::AND) { for (const Term& c : t) conjuncts(c, out); }
        else if (!(t.getKind() == Kind::CONST_BOOLEAN && t.getBooleanValue())) out.push_back(t);
    }
    static bool defines(const Term& a, Term& c) {
        if (a.getKind() != Kind::EQUAL || a.getNumChildren() != 2) return false;
        for (int side = 0; side < 2; side++)
            if (a[side].getKind() == Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV && a[side][0].getKind() == Kind::CONSTANT) {
                c = a[side][0];
                return true;
            }
        return false;
    }
    std::vector<Term> cone(const std::vector<Term>& as, size_t skip, const Term& goal) {
        std::unordered_map<uint64_t, std::vector<size_t>> defs;
        std::vector<char> take(as.size(), 0);
        std::vector<Term> need;
        std::unordered_set<uint64_t> seen;
        auto consts = [&](const Term& t) {
            std::vector<Term> st{t};
            while (!st.empty()) {
                Term x = st.back(); st.pop_back();
                if (!seen.insert(x.getId()).second) continue;
                if (x.getKind() == Kind::CONSTANT) need.push_back(x);
                for (const Term& y : x) st.push_back(y);
            }
        };
        for (size_t i = 0; i < as.size(); i++) {
            if (i == skip) continue;
            Term c;
            if (defines(as[i], c)) defs[c.getId()].push_back(i);
            else { take[i] = 1; consts(as[i]); }         // assumptions always stay
        }
        consts(goal);
        for (size_t k = 0; k < need.size(); k++) {
            auto it = defs.find(need[k].getId());
            if (it == defs.end()) continue;
            for (size_t i : it->second) if (!take[i]) { take[i] = 1; consts(as[i]); }
        }
        std::vector<Term> out;
        for (size_t i = 0; i < as.size(); i++) if (take[i]) out.push_back(as[i]);
        out.push_back(goal);
        return out;
    }
    // A small-domain value used as a shift amount (vadd's shift in [0, 31]) is a
    // barrel shifter per lane on each side. Case-split it instead: each case
    // substitutes one value, which makes every shift by it a constant shift. The
    // assumptions put the value in [lo, hi] in every model, so the cases are
    // exhaustive and all of them Unsat is a proof.
    CheckResult check_enum_shift(const std::vector<Term>& original, uint64_t ms) {
        std::unordered_map<uint64_t, BvR> seeds = bv_seeds(original);
        std::unordered_set<uint64_t> seen, in_amount;
        std::vector<Term> stack(original.begin(), original.end()), amounts;
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!seen.insert(t.getId()).second) continue;
            Kind k = t.getKind();
            if ((k == Kind::BITVECTOR_SHL || k == Kind::BITVECTOR_LSHR || k == Kind::BITVECTOR_ASHR) &&
                t[1].getKind() != Kind::CONST_BITVECTOR)
                amounts.push_back(t[1]);
            for (const Term& c : t) stack.push_back(c);
        }
        for (size_t i = 0; i < amounts.size(); i++) {
            if (!in_amount.insert(amounts[i].getId()).second) continue;
            for (const Term& c : amounts[i]) amounts.push_back(c);
        }
        Term x;
        __int128 lo = 0, hi = -1;
        for (const auto& [id, r] : seeds) {
            if (!in_amount.count(id)) continue;
            const BvIv& S = r.first;
            if (S.hi - S.lo + 1 > 32 || S.hi < S.lo || (!x.isNull() && S.hi - S.lo >= hi - lo)) continue;
            for (const Term& a : original)          // the seeded term itself (seeds are keyed by id)
                if (find_by_id(a, id, x)) break;
            lo = S.lo; hi = S.hi;
        }
        if (x.isNull()) return CheckResult::Unknown;
        uint32_t w = x.getSort().getBitVectorSize();
        auto t0 = std::chrono::steady_clock::now();
        for (__int128 v = lo; v <= hi; v++) {
            Term val = bv_val(w, (uint64_t)v & (w >= 64 ? ~0ULL : ((1ULL << w) - 1)));
            std::vector<Term> cs;
            for (const Term& a : original) cs.push_back(a.substitute(x, val));
            cs.push_back(tm.mkTerm(Kind::EQUAL, {x, val}));
            CheckResult r = uf_first_ops_.empty() ? CheckResult::Unknown : check_uf_first(cs);
            if (r != CheckResult::Unsat) {
                uint64_t used = (uint64_t)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                uint64_t left = ms > used ? ms - used : 1000;
                r = check_simplified(cs, ms ? std::max<uint64_t>(1000, left / (uint64_t)(hi - v + 1)) : 0);
            }
            if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1')
                std::fprintf(stderr, "[enum] %s = %lld: %s\n", x.toString().substr(0, 60).c_str(), (long long)v,
                             r == CheckResult::Unsat ? "unsat" : "not unsat");
            if (r != CheckResult::Unsat) return CheckResult::Unknown;
        }
        return CheckResult::Unsat;
    }
    static bool find_by_id(const Term& t, uint64_t id, Term& out) {
        std::vector<Term> st{t};
        std::unordered_set<uint64_t> vis;
        while (!st.empty()) {
            Term u = st.back(); st.pop_back();
            if (u.getId() == id) { out = u; return true; }
            if (!vis.insert(u.getId()).second) continue;
            for (const Term& c : u) st.push_back(c);
        }
        return false;
    }
    CheckResult check_split(const std::vector<Term>& original, std::chrono::steady_clock::time_point t0) {
        if (original.empty() || original.back().getKind() != Kind::NOT) return CheckResult::Unknown;
        std::vector<Term> cases;
        conjuncts(original.back()[0], cases);
        // A single output is split too, for the shift enumeration and the proof cache below.
        const bool paper = true;
        if (cases.empty() || cases.size() > 4096) return CheckResult::Unknown;
        size_t done = 0, cached = 0;
        for (const Term& c : cases) {
            std::vector<Term> sliced = cone(original, original.size() - 1, tm.mkTerm(Kind::NOT, {c}));
            if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1')
                std::fprintf(stderr, "[split] output %zu/%zu: %zu of %zu assertions\n",
                             done + 1, cases.size(), sliced.size(), original.size());
            std::string key;
            if (paper) {
                key = canonical_key(sliced);
                if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1')
                    std::fprintf(stderr, "[split] key %zx size %zu cached %d\n", std::hash<std::string>{}(key), key.size(),
                                 (int) unsat_cache().count(key));
                if (const char* kd = std::getenv("SALT_KEY_DUMP"); kd && *kd) {
                    static int nk = 0;
                    if (FILE* f = std::fopen((std::string(kd) + "/key" + std::to_string(nk++) + ".txt").c_str(), "w")) {
                        std::fwrite(key.data(), 1, key.size(), f); std::fclose(f); }
                }
                if (unsat_cache().count(key)) { done++; cached++; continue; }
            }
            uint64_t ms = query_timeout_ms_ ? std::max<uint64_t>(1000, remaining_ms(t0) / (cases.size() - done)) : 0;
            CheckResult r = CheckResult::Unknown;
            // SALT_BWZ_FIRST=1: an integer-only output goes to Bitwuzla first. Its Unsat is a proof of this
            // output's exact sliced query, so it enters the cache like any other stage's.
            if (bwz_lanes() && !terms_have_fp(sliced) && solve_external(std::getenv("SALT_BITWUZLA"), ms, &sliced) ==
                                                              CheckResult::Unsat)
                r = CheckResult::Unsat, bwz_lane_proofs_++;
            if (r != CheckResult::Unsat && paper) r = check_enum_shift(sliced, ms);
            // A single output has nothing to split: past the cache and the shift
            // enumeration the main pipeline decides it (with its integer-sum stage,
            // which rsum needs).
            if (r != CheckResult::Unsat && cases.size() == 1) return CheckResult::Unknown;
            if (r != CheckResult::Unsat) {
                std::vector<Term> as = interval_simplified(sliced);
                size_t merged = 0;
                std::vector<Term> lat = lattice_normalized(as, merged, &sliced);
                if (!uf_first_ops_.empty()) r = check_uf_first(merged ? lat : as);
                if (r != CheckResult::Unsat && merged) r = check_unsat_only(lat, ms);
                if (r != CheckResult::Unsat && !std::getenv("SALT_NO_AC")) {   // max/min trees regrouped (any NaN)
                    size_t acn = 0;
                    std::vector<Term> ac = ac_minmax_normalized(as, acn);
                    if (acn) r = check_unsat_only(ac, ms);
                    if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1' && acn)
                        std::fprintf(stderr, "[ac] %zu max/min rewrites: %s\n", acn, r == CheckResult::Unsat ? "unsat" : "not unsat");
                }
                if (r != CheckResult::Unsat) r = check_simplified(as, ms);
            }
            if (r != CheckResult::Unsat) return CheckResult::Unknown;   // the full query decides (and gives a model)
            if (paper) unsat_cache().insert(key);
            done++;
        }
        if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1')
            std::fprintf(stderr, "[split] %zu outputs, %zu from the proof cache\n", cases.size(), cached);
        return CheckResult::Unsat;
    }

    // An output's sliced query with its free constants renamed in order of first
    // occurrence. Lanes on one path of an elementwise kernel differ only in which
    // input bytes they read, so their queries print the same; renaming free
    // constants preserves satisfiability, so one Unsat covers every lane with
    // that key, in this query and in the longer ones after it (the cache is per
    // process; each batch has its own context and term manager).
    std::map<std::string, Term> canon_pool_;
    std::string canonical_key(const std::vector<Term>& as) {
        std::vector<Term> olds, news;
        std::unordered_set<uint64_t> vis;
        std::map<std::string, size_t> per_sort;
        for (const Term& a : as) {
            std::vector<Term> st{a};
            while (!st.empty()) {
                Term t = st.back(); st.pop_back();
                if (!vis.insert(t.getId()).second) continue;
                if (t.getKind() == Kind::CONSTANT) {
                    std::string sort = t.getSort().toString();
                    std::string name = "k" + std::to_string(per_sort[sort]++) + "_" + sort;
                    auto it = canon_pool_.find(name);
                    if (it == canon_pool_.end()) it = canon_pool_.emplace(name, tm.mkConst(t.getSort(), name)).first;
                    olds.push_back(t); news.push_back(it->second);
                    continue;
                }
                std::vector<Term> ch(t.begin(), t.end());          // preorder, children left to right
                for (auto c = ch.rbegin(); c != ch.rend(); ++c) st.push_back(*c);
            }
        }
        std::string key;
        for (const Term& a : as) key += (olds.empty() ? a : a.substitute(olds, news)).toString() + "\n";
        return key;
    }
    size_t bwz_lane_proofs_ = 0;
    static bool bwz_lanes() {
        const char* f = std::getenv("SALT_BWZ_FIRST"); const char* b = std::getenv("SALT_BITWUZLA");
        return f && std::string(f) == "1" && b && *b;
    }
    static std::unordered_set<std::string>& unsat_cache() {
        static std::unordered_set<std::string> cache;
        return cache;
    }

    uint64_t stage_ms(double share) const {
        if (!query_timeout_ms_) return 60000;
        return std::max<uint64_t>(1000, (uint64_t)(query_timeout_ms_ * share));
    }

    // The simplified exact query in a fresh solver. It has exactly the original's
    // models, so its answer stands; only Sat goes back to the original solver,
    // for a model the witness code can read.
    CheckResult check_simplified(const std::vector<Term>& as, uint64_t ms) {
        Solver s3(tm);
        s3.setOption("fp-exp", "true");
        if (ms) s3.setOption("tlimit-per", std::to_string(ms));
        for (const Term& a : as) s3.assertFormula(a);
        Result r = s3.checkSat();
        return r.isUnsat() ? CheckResult::Unsat : r.isSat() ? CheckResult::Sat : CheckResult::Unknown;
    }
    uint64_t remaining_ms(std::chrono::steady_clock::time_point t0) const {
        if (!query_timeout_ms_) return 0;
        uint64_t used = (uint64_t)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return query_timeout_ms_ > used + 1000 ? query_timeout_ms_ - used : 1000;
    }

    // Solve the whole assertion stack with uf_first_ops_ abstracted, in a second
    // solver on the same TermManager. Only an Unsat answer is meaningful.
    CheckResult check_uf_first(const std::vector<Term>& as) {
        CheckResult r = check_uf_first_on(as);
        if (r == CheckResult::Unsat || std::getenv("SALT_NO_RING")) return r;
        // Not refuted as written: once more on the ring normal form (same models), where a folded bias and an
        // unfolded one abstract to the same products.
        size_t before = ring_rewrites_;
        std::vector<Term> nf = ring_normal(as);
        if (ring_rewrites_ == before) return r;
        if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1')
            std::fprintf(stderr, "[ring] %zu rewrites\n", ring_rewrites_ - before);
        return check_uf_first_on(nf);
    }
    CheckResult check_uf_first_on(const std::vector<Term>& as) {
        Solver s2(tm);
        s2.setOption("fp-exp", "true");
        // When the abstraction fits, the UF query is fast; when it does not, time
        // spent here is lost, so the stage gets a quarter of the budget, at most 60 s.
        uint64_t ms = std::min<uint64_t>(stage_ms(0.25), 60000);
        if (const char* m = std::getenv("SALT_UF_FIRST_MS"); m && *m) ms = std::stoull(m);
        s2.setOption("tlimit-per", std::to_string(ms));
        std::string saved = uf_ops_;
        uf_ops_ = uf_first_ops_;
        std::unordered_map<uint64_t, Term> memo;
        std::vector<Term> rw;
        for (const Term& a : as) rw.push_back(uf_rewrite_rec(a, memo));
        uf_ops_ = saved;
        size_t n_uf = 0;
        for (const auto& kv : memo) n_uf += kv.second.getKind() == Kind::APPLY_UF;
        if (n_uf == 0) return CheckResult::Unknown;    // nothing abstracted: the exact query decides
        // Interval axioms: seed from the assumptions, push through FP equalities
        // (store_fp32_as_bv's to_fp(bv) == value) to a fixed point, then bound
        // every abstracted application whose operands are bounded.
        std::unordered_map<uint64_t, Iv> seeds;
        std::vector<std::pair<Term, Term>> eqs;
        std::unordered_map<uint64_t, std::optional<Iv>> ivs;
        interval_state(as, seeds, eqs, ivs);
        // Bounds matter only where a UF result meets interpreted logic (a clamp,
        // a comparison, the stored-output equality); an argument of another UF
        // needs none. Bound only those frontier applications.
        // A store's definitional equality to_fp(fresh) = u (every lane result
        // becomes bits) is a frontier only if the fresh bits are used elsewhere: a
        // compared output or a bit manipulation, not a register-only intermediate.
        std::unordered_map<uint64_t, size_t> refs;
        std::unordered_set<uint64_t> frontier, seen;
        std::vector<Term> stack, nodes;
        for (const Term& a : rw) stack.push_back(a);
        while (!stack.empty()) {
            Term t = stack.back(); stack.pop_back();
            if (!seen.insert(t.getId()).second) continue;
            nodes.push_back(t);
            for (const Term& c : t) { refs[c.getId()]++; stack.push_back(c); }
        }
        auto definitional = [&](const Term& eq, Term& u) {
            if (eq.getKind() != Kind::EQUAL || eq.getNumChildren() != 2) return false;
            for (int side = 0; side < 2; side++) {
                Term x = eq[side], y = eq[1 - side];
                if (x.getKind() == Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV &&
                    x[0].getKind() == Kind::CONSTANT && y.getKind() == Kind::APPLY_UF) {
                    if (refs[x.getId()] > 1 || refs[x[0].getId()] > 1) return false;   // bits used elsewhere
                    u = y;
                    return true;
                }
            }
            return false;
        };
        for (const Term& t : nodes) {
            if (t.getKind() == Kind::APPLY_UF) continue;
            Term u;
            if (definitional(t, u)) continue;
            for (const Term& c : t)
                if (c.getKind() == Kind::APPLY_UF) frontier.insert(c.getId());
        }
        // A dead definitional equality constrains nothing else; leaving it out
        // only weakens the abstract query, so its UNSAT still implies ours.
        // Likewise an input band whose subjects feed only UFs: its content is in
        // the interval axioms already. Subjects used by interpreted FP logic keep it.
        auto range_atom = [](const Term& t) {
            Kind k = t.getKind();
            if ((k != Kind::FLOATINGPOINT_LEQ && k != Kind::FLOATINGPOINT_GEQ &&
                 k != Kind::FLOATINGPOINT_LT && k != Kind::FLOATINGPOINT_GT) || t.getNumChildren() != 2)
                return false;
            double v;
            return fp32_const(t[0], v) || fp32_const(t[1], v);
        };
        std::unordered_set<uint64_t> interpreted_use;
        for (const Term& t : nodes) {
            if (t.getKind() == Kind::APPLY_UF || range_atom(t)) continue;
            for (const Term& c : t) interpreted_use.insert(c.getId());
        }
        auto redundant_band = [&](const Term& f) {
            std::vector<Term> atoms{f};
            for (size_t i = 0; i < atoms.size(); i++) {
                Term a = atoms[i];
                if (a.getKind() == Kind::AND) { for (const Term& c : a) atoms.push_back(c); continue; }
                if (!range_atom(a)) return false;
                double v;
                Term subj = fp32_const(a[1], v) ? a[0] : a[1];
                if (interpreted_use.count(subj.getId())) return false;
            }
            return true;
        };
        size_t dropped = 0, bands = 0;
        for (const Term& a : rw) {
            Term u;
            if (definitional(a, u)) { dropped++; continue; }
            if (redundant_band(a)) { bands++; continue; }
            s2.assertFormula(a);
        }
        size_t axioms = 0;
        std::unordered_map<uint64_t, Term> fp_by_id;      // the floating-point terms of the query, by id
        {
            std::vector<Term> st(as.begin(), as.end());
            std::unordered_set<uint64_t> vis;
            while (!st.empty()) {
                Term t = st.back(); st.pop_back();
                if (!vis.insert(t.getId()).second) continue;
                if (t.getSort().isFloatingPoint()) fp_by_id.emplace(t.getId(), t);
                for (const Term& c : t) st.push_back(c);
            }
        }
        for (const auto& [id, rew] : memo) {
            if (rew.getKind() != Kind::APPLY_UF || !frontier.count(rew.getId())) continue;
            auto iv = ivs.find(id);
            if (iv == ivs.end() || !iv->second) {
                // interval_state evaluates the terms it reaches through the stored-float equalities; a sliced query
                // has the float arithmetic inline and none of those, so the application is evaluated here.
                auto o = fp_by_id.find(id);
                Iv direct;
                if (o == fp_by_id.end() || !iv_of(o->second, seeds, ivs, direct)) continue;
                iv = ivs.find(id);
                if (iv == ivs.end() || !iv->second) continue;
            }
            s2.assertFormula(tm.mkTerm(Kind::AND, {
                tm.mkTerm(Kind::FLOATINGPOINT_LEQ, {fp32_bound(iv->second->lo, false), rew}),
                tm.mkTerm(Kind::FLOATINGPOINT_LEQ, {rew, fp32_bound(iv->second->hi, true)})}));
            axioms++;
        }
        // Abstracted integer products: the real product lies in its operands'
        // range product wherever that cannot wrap, in every model of the
        // assumptions, so the bound keeps the abstraction an over-approximation.
        if (uf_first_ops_.find("bvmul") != std::string::npos) {
            std::unordered_map<uint64_t, BvR> seeds = bv_seeds(as), bmemo;
            std::vector<Term> st(as.begin(), as.end());
            std::unordered_set<uint64_t> vis;
            while (!st.empty()) {
                Term t = st.back(); st.pop_back();
                if (!vis.insert(t.getId()).second) continue;
                for (const Term& c : t) st.push_back(c);
                if (t.getKind() != Kind::BITVECTOR_MULT) continue;
                auto m = memo.find(t.getId());
                if (m == memo.end() || m->second.getKind() != Kind::APPLY_UF) continue;
                const Term& u = m->second;
                uint32_t w = t.getSort().getBitVectorSize();
                if (w > 64) continue;
                auto [S, U] = bv_range_in(t, seeds, bmemo);
                auto [fs, fu] = bv_full(w);
                if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1') {
                    std::fprintf(stderr, "[uf-first] %u-bit product: signed [%lld, %lld]\n", w, (long long)S.lo, (long long)S.hi);
                    for (const Term& c : t) {
                        BvIv cs = bv_range_in(c, seeds, bmemo).first;
                        std::fprintf(stderr, "[uf-first]   operand signed [%lld, %lld]: %.300s\n", (long long)cs.lo,
                                     (long long)cs.hi, c.toString().c_str());
                    }
                }
                auto val = [&](__int128 x) { return bv_val(w, (uint64_t)x & (w >= 64 ? ~0ULL : ((1ULL << w) - 1))); };
                // A product with a zero factor is zero (also mod 2^w). Without this, two kernels that pick different
                // multipliers for a zero operand (x*m_neg against x*m_pos at x = 0) differ under the abstraction.
                if (!std::getenv("SALT_NO_UF_ZERO")) {
                    // ... and it commutes: the abstraction orders a product's operands by term id, which matches
                    // a*b with b*a only when the operands are the same terms, not merely equal ones.
                    if (u.getNumChildren() == 3) {
                        s2.assertFormula(tm.mkTerm(Kind::EQUAL, {u, tm.mkTerm(Kind::APPLY_UF, {u[0], u[2], u[1]})}));
                        axioms++;
                    }
                    for (const Term& c : t) {
                        auto rc = memo.find(c.getId());
                        const Term& cr = rc != memo.end() ? rc->second : c;
                        s2.assertFormula(tm.mkTerm(Kind::IMPLIES, {
                            tm.mkTerm(Kind::EQUAL, {cr, bv_val(cr.getSort().getBitVectorSize(), 0)}),
                            tm.mkTerm(Kind::EQUAL, {u, val(0)})}));
                        axioms++;
                    }
                }
                if (S.lo > fs.lo || S.hi < fs.hi) {
                    s2.assertFormula(tm.mkTerm(Kind::AND, {tm.mkTerm(Kind::BITVECTOR_SGE, {u, val(S.lo)}),
                                                           tm.mkTerm(Kind::BITVECTOR_SLE, {u, val(S.hi)})}));
                    axioms++;
                }
                if (U.lo > fu.lo || U.hi < fu.hi) {
                    s2.assertFormula(tm.mkTerm(Kind::AND, {tm.mkTerm(Kind::BITVECTOR_UGE, {u, val(U.lo)}),
                                                           tm.mkTerm(Kind::BITVECTOR_ULE, {u, val(U.hi)})}));
                    axioms++;
                }
            }
        }
        uf_axioms_ = axioms;
        auto t0 = std::chrono::steady_clock::now();
        Result r = s2.checkSat();
        if (const char* d = std::getenv("SALT_UF_DEBUG"); d && *d == '1') {
            size_t abstracted = 0;
            for (const auto& kv : memo) abstracted += kv.second.getKind() == Kind::APPLY_UF;
            std::fprintf(stderr, "[uf-first] %zu abstracted ops, %zu interval axioms, %zu seeds, %zu eqs, %zu dead stores and %zu bands dropped: %s in %.2fs\n",
                         abstracted, axioms, seeds.size(), eqs.size(), dropped, bands, r.toString().c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return r.isUnsat() ? CheckResult::Unsat : CheckResult::Unknown;
    }
    const char* method() const { return method_; }

    CheckResult check() {
        dump_smt2_if_requested();
        method_ = "smt";
        auto t0 = std::chrono::steady_clock::now();
        bool simplify = !std::getenv("SALT_SIMPLIFY") || std::string(std::getenv("SALT_SIMPLIFY")) != "0";
        const bool fp = query_has_fp();
        if ((simplify || !uf_first_ops_.empty()) && (fp || simplify)) {
            const std::vector<Term> original = solver->getAssertions();
            std::vector<Term> as = original;
            if (simplify) as = interval_simplified(as);
            // The integer normal form (regrouped sums) is its own stage, for BV
            // queries: it decides rdsum's at once, while on an FP kernel's integer
            // part (qc4w) it settled nothing and cost its share.
            std::vector<Term> ints;
            if (simplify && !fp && !std::getenv("SALT_NO_INT_SUMS")) {
                size_t before = int_rewrites_;
                ints = interval_simplified(original, true);
                if (int_rewrites_ == before) ints.clear();
            }
            // One output at a time first: refuting every lane's miter at once grows
            // far faster than their number (15 s for 32 qs8-vcvt lanes with their
            // products abstracted; f32-vrndne reached 36 outputs in 10 minutes).
            // Under either NaN policy, so the multidimensional kernels share it.
            if (simplify && check_split(original, t0) == CheckResult::Unsat) {
                method_ = bwz_lane_proofs_ ? "split+bwz" : "split";
                return CheckResult::Unsat;
            }
            // The lattice form feeds the over-approximating stages only.
            size_t lattice = 0;
            std::vector<Term> lat;
            if (simplify && !std::getenv("SALT_NO_LATTICE")) lat = lattice_normalized(as, lattice);
            if (!uf_first_ops_.empty() && check_uf_first(lattice ? lat : as) == CheckResult::Unsat) {
                method_ = lattice ? "uf+lattice" : "uf";
                return CheckResult::Unsat;
            }
            if (lattice && check_unsat_only(lat, query_timeout_ms_ ? std::min(stage_ms(0.5), remaining_ms(t0)) : 0) ==
                               CheckResult::Unsat) {
                method_ = "lattice";
                return CheckResult::Unsat;
            }
            if (simplify && !std::getenv("SALT_NO_AC")) {      // max/min trees regrouped (any NaN); Unsat only
                size_t acn = 0;
                std::vector<Term> ac = ac_minmax_normalized(as, acn);
                if (acn && check_unsat_only(ac, query_timeout_ms_ ? std::min(stage_ms(0.5), remaining_ms(t0)) : 0) ==
                               CheckResult::Unsat) {
                    method_ = "ac";
                    return CheckResult::Unsat;
                }
            }
            if (!ints.empty()) {                           // same models: Unsat is a proof
                // Half the budget: a regrouped query can be harder to refute, so the
                // original keeps the rest (and supplies counterexample models).
                uint64_t ms = query_timeout_ms_ ? std::max<uint64_t>(1000, remaining_ms(t0) / 2) : 0;
                if (check_simplified(ints, ms) == CheckResult::Unsat) {
                    method_ = "int";
                    return CheckResult::Unsat;
                }
            }
            if (simplify && fp) {
                CheckResult r = check_simplified(as, remaining_ms(t0));
                if (r == CheckResult::Unsat) { method_ = "simp"; return CheckResult::Unsat; }
                if (r == CheckResult::Unknown) return CheckResult::Unknown;   // same models: a timeout stands
            }
            if (query_timeout_ms_)                     // the exact query gets what is left
                solver->setOption("tlimit-per", std::to_string(remaining_ms(t0)));
        }
    exact:
        const char* mode = std::getenv("SALT_BACKEND");
        const char* bwz  = std::getenv("SALT_BITWUZLA");
        bool have_bwz = bwz && *bwz;
        if (!mode || std::string(mode) != "auto" || !have_bwz)
            return check_cvc5();

        if (query_has_fp()) {                       // FP → cvc5 primary
            CheckResult r = check_cvc5();
            if (r != CheckResult::Unknown) return r;          // definitive (model on Sat)
            // cvc5 timed out — bitwuzla rarely helps on FP, but try for an Unsat.
            if (solve_external(bwz) == CheckResult::Unsat) return CheckResult::Unsat;
            return CheckResult::Unknown;            // no cvc5 model for a Sat → honest TIMEOUT
        }
        // pure BV → bitwuzla primary
        CheckResult rb = solve_external(bwz);
        if (rb == CheckResult::Unsat) return CheckResult::Unsat;  // fast verified — the win
        return check_cvc5();                        // Sat/Unknown → cvc5 for model + fallback
    }

    // Base-10 BV value as string (for CEX printing).
    std::string value_str(Term t) {
        return solver->getValue(t).getBitVectorValue(10);
    }

    // Per-query timeout in milliseconds.  Applied to cvc5 (tlimit-per) and, when
    // routing, passed to the external solver via its own -t flag.
    uint64_t query_timeout_ms_ = 0;
    void set_query_timeout_ms(uint64_t ms) {
        query_timeout_ms_ = ms;
        solver->setOption("tlimit-per", std::to_string(ms));
    }
};

// ---------------------------------------------------------------------------
// Global thread-local context pointer
// ---------------------------------------------------------------------------
inline thread_local VerificationContext* g_ctx = nullptr;

// Scalar-term side channel for salt_float16's copy-ctor (see symbolic_buffer.hpp).
inline thread_local std::unordered_map<const void*, Term> g_scalar_terms;

// FP<->BV memoization: store_fp{16,32,64}_as_bv mints a fresh BV `bv` with a global
// `TO_FP_FROM_IEEE_BV(bv) == fp` constraint; we map bv.getId() -> fp so load_as_fp*(bv)
// returns the FP term DIRECTLY instead of re-introducing TO_FP_FROM_IEEE_BV(bv). For pure
// FP-op chains the fresh bv is then read only by its own constraint (dead, pruned by cvc5);
// it stays live only where the bits are genuinely manipulated (vreinterpret bitcasts) or
// stored to a buffer. Collapses the per-op fresh-var blowup that made exp/conv chains
// intractable. Keyed by Term id (matching the uf_rewrite memo convention).
inline thread_local std::unordered_map<uint64_t, Term> g_fp_bv_cache;

inline void VerificationContext::clear_scalar_terms() { g_scalar_terms.clear(); g_fp_bv_cache.clear(); }
inline VerificationContext::~VerificationContext() { g_scalar_terms.clear(); g_fp_bv_cache.clear(); }

// ---------------------------------------------------------------------------
// Helper: BV value from signed int64, masked to width
// ---------------------------------------------------------------------------
inline Term mk_bv_val(TermManager& tm, size_t bits, int64_t val) {
    uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1);
    return tm.mkBitVector(static_cast<uint32_t>(bits),
                          static_cast<uint64_t>(val) & mask);
}

// ---------------------------------------------------------------------------
// FP16 from concrete salt_float16, possibly redirected through scalar side channel
// ---------------------------------------------------------------------------
inline Term mk_fp16_from_f16(TermManager& tm, const salt_float16& value) {
    auto it = g_scalar_terms.find(&value);
    if (it != g_scalar_terms.end()) return it->second;
    Term bv = tm.mkBitVector(16, value.value);
    Op op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
    return tm.mkTerm(op, {bv});
}

} // namespace salt_cvc5
