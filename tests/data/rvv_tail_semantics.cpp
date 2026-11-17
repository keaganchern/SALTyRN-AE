// Element-level checks of the RVV models where vl, VLMAX and tail elements meet (tests/test_rvv_tail_semantics.py).
#include "salt.hpp"
#include <cstdio>
using namespace salt;
static int fails = 0;
// The value a term takes in every model: solve twice, once asking for a different value.
static bool fixed_value(VerificationContext& ctx, const Term& t, uint64_t want) {
    auto& tm = ctx.tm;
    Term ne = tm.mkTerm(Kind::NOT, {tm.mkTerm(Kind::EQUAL, {t, tm.mkBitVector(t.getSort().getBitVectorSize(), want)})});
    ctx.solver->push(); ctx.solver->assertFormula(ne); bool sat = ctx.solver->checkSat().isSat(); ctx.solver->pop();
    return !sat;
}
static bool can_differ(VerificationContext& ctx, const Term& t, uint64_t v) { return !fixed_value(ctx, t, v); }
#define CHECK(c, msg) do { if (!(c)) { fails++; printf("FAIL %s\n", msg); } else printf("ok   %s\n", msg); } while (0)
int main() {
    VerificationContext ctx(128); g_ctx = &ctx;
    auto& tm = ctx.tm;
    std::vector<Term> e; for (int i = 0; i < 16; i++) e.push_back(tm.mkBitVector(8, 10 + i));
    RVVVector full(tm, 8, e);                                            // u8m1 at VLEN 128: VLMAX 16
    // vslidedown by 2 with vl 4: lanes read src[2..5] (past vl, below VLMAX), not 0
    RVVVector d = __riscv_vslidedown_vx_u8m2(RVVVector(tm, 8, std::vector<Term>(32, tm.mkBitVector(8, 7))), 30, 4);
    CHECK(fixed_value(ctx, d.getElement(0), 7) && fixed_value(ctx, d.getElement(1), 7) && fixed_value(ctx, d.getElement(2), 0),
          "u8m2 slidedown 30 (VLMAX 32): lanes 0,1 = src[30,31], lane 2 past VLMAX = 0");
    RVVVector g = __riscv_vrgather_vx_u8m1(full, 9, 4);
    CHECK(fixed_value(ctx, g.getElement(0), 19), "vrgather_vx index 9 with vl 4 reads src[9]");
    RVVVector g2 = __riscv_vrgather_vx_u8m1(full, 16, 4);
    CHECK(fixed_value(ctx, g2.getElement(0), 0), "vrgather_vx index 16 (= VLMAX) reads 0");
    // a short source (produced with vl 4): its lanes 4.. are tail, unspecified
    RVVVector shortv(tm, 8, std::vector<Term>(e.begin(), e.begin() + 4));
    RVVVector g3 = __riscv_vrgather_vx_u8m1(shortv, 6, 4);
    CHECK(can_differ(ctx, g3.getElement(0), 0) && can_differ(ctx, g3.getElement(0), 16), "gather of a short source's tail lane is unconstrained");
    // f32m4 slidedown (hand-written before): VLEN 128, VLMAX 16; slide 3 with vl 4 reads src[3..6]
    { std::vector<Term> f; for (int i = 0; i < 16; i++) f.push_back(tm.mkBitVector(32, 100 + i));
      RVVVector a(tm, 32, f), d4 = __riscv_vslidedown_vx_f32m4(a, 3, 4);
      CHECK(fixed_value(ctx, d4.getElement(3), 106), "f32m4 slidedown 3, vl 4: lane 3 = src[6]");
      RVVVector gi(tm, 32, std::vector<Term>(4, tm.mkBitVector(32, 9)));
      RVVVector g9 = __riscv_vrgather_vv_f32m2(RVVVector(tm, 32, std::vector<Term>(f.begin(), f.begin() + 8)), gi, 4);
      CHECK(fixed_value(ctx, g9.getElement(0), 0), "f32m2 gather index 9 (VLMAX 8) reads 0");
      RVVVector gi6(tm, 32, std::vector<Term>(4, tm.mkBitVector(32, 6)));
      RVVVector g6 = __riscv_vrgather_vv_f32m2(RVVVector(tm, 32, std::vector<Term>(f.begin(), f.begin() + 4)), gi6, 4);
      CHECK(can_differ(ctx, g6.getElement(0), 0) && can_differ(ctx, g6.getElement(0), 106),
            "f32m2 gather index 6 of a source produced with vl 4: a tail element, unconstrained");
    }
    { RVVVector m = __riscv_vmv_s_x_i32m1(5, 4);
      CHECK(fixed_value(ctx, m.getElement(0), 5) && can_differ(ctx, m.getElement(1), 0), "vmv_s_x: element 0 set, element 1 unspecified"); }
    RVVVector u = __riscv_vundefined_u8m1();
    CHECK(can_differ(ctx, u.getElement(0), 0), "vundefined lane is unconstrained");
    bool threw = false;
    try { (void) shortv.getElement(5); } catch (const NativeAccess& x) { threw = true; printf("     %s\n", x.what()); }
    CHECK(threw, "reading a short vector's tail element refuses");
    printf(fails ? "FAILED %d\n" : "ALL OK\n", fails);
    return fails != 0;
}
