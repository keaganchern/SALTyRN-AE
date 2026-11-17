"""Verifier fixes from the false-failure hunt: a correct kernel the engine could not verify.

- Weight-layout inference sees only the kernel's own code (the entry function and the helpers it calls), not an
  unused helper such as a benchmark's layout converter, whose vsetvl LMUL made it refuse.
- A counterexample can be dumped (SALT_CEX_JSON) for an independent concrete replay; the dump is emitted on the
  counterexample path only."""
from src.workflow.verification.engine.emit import _kernel_only, _rvv_chunked_weights, _rvv_padded_tile

KERNEL = """
static inline vfloat32m4_t twice(vfloat32m4_t v, size_t vl) { return __riscv_vfadd_vv_f32m4(v, v, vl); }
void test_rvv(size_t channels, const float* w, float* out) {
  for (size_t c = channels; c > 0; ) {
    const size_t vl = __riscv_vsetvl_e32m4(c);
    __riscv_vse32_v_f32m4(out, twice(__riscv_vle32_v_f32m4(w, vl), vl), vl);
    w += vl; out += vl; c -= vl;
  }
}
"""
CONVERTER = """
void pack_weights_from_m8(const void* src, size_t channels, void* dst, size_t cap) {
  const size_t vl = __riscv_vsetvl_e32m8(channels);
  __riscv_vse32_v_f32m8((float*) dst, __riscv_vle32_v_f32m8((const float*) src, vl), vl);
}
"""


def test_unused_helper_does_not_steer_the_layout():
    # The converter's m8 used to make the inference see two LMULs and refuse.
    assert _rvv_chunked_weights(KERNEL + CONVERTER) == "(vlen / 32) * 4"
    assert _rvv_chunked_weights(CONVERTER + KERNEL) == "(vlen / 32) * 4"


def test_called_helpers_are_kept():
    only = _kernel_only(KERNEL + CONVERTER)
    assert "twice(" in only and "test_rvv(" in only and "pack_weights_from_m8" not in only


def test_a_second_lmul_in_the_kernel_itself_is_still_refused():
    # Mixed LMULs inside the kernel can change the chunk sequence; that stays a refusal, not a guess.
    mixed = KERNEL.replace("__riscv_vfadd_vv_f32m4(v, v, vl)", "__riscv_vfadd_vv_f32m4(v, v, __riscv_vsetvl_e32m8(vl))")
    try:
        _rvv_chunked_weights(mixed)
    except NotImplementedError:
        return
    raise AssertionError("mixed LMULs in the kernel were not refused")


def test_source_without_the_entry_is_unchanged():
    assert _kernel_only(CONVERTER, entry="test_rvv") == CONVERTER


def test_padded_tile_ignores_unused_helpers():
    gemm = """
void test_rvv(size_t nc, const float* w, float* c) {
  const size_t nr = __riscv_vsetvlmax_e32m4();
  for (; nc != 0; nc -= nr < nc ? nr : nc) { w += nr; }
}
"""
    helper = "void unused(void) { const size_t nr = __riscv_vsetvlmax_e32m8(); (void) nr; }\n"
    assert _rvv_padded_tile(helper + gemm) == "(vlen / 32) * 4"
