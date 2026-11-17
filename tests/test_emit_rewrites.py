"""Scalar reads of symbolic integer inputs must reach the symbolic buffer."""
from src.workflow.verification.engine.emit import _native_reads, _rewrite_index_reads

KERNEL = """void test_rvv(size_t kc, const int8_t* a0, float* c) {
  const int8_t* ak = a0;
  for (size_t k = 0; k < kc; k += 2) {
    const int8_t a_v0 = ak[0];
    const int8_t a_v1 = ak[1];
    ak += 2;
  }
}
"""


def test_a_copied_pointer_is_followed():
    assert _native_reads(KERNEL, ["a0"]) == "a0 via ak"
    out = _rewrite_index_reads(KERNEL, ["a0"])
    assert "salt_scalar(ak + (0))" in out and "const auto a_v1" in out
    assert _native_reads(out, ["a0"]) is None


def test_redeclared_locals_in_two_scopes_are_both_rebound():
    twice = KERNEL.replace("    ak += 2;\n  }\n", "    ak += 2;\n  }\n  if (kc) {\n    const int8_t a_v0 = ak[0];\n  }\n")
    out = _rewrite_index_reads(twice, ["a0"])
    assert "const int8_t a_v0" not in out
