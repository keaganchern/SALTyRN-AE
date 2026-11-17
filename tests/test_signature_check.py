"""The RVV kernel must keep the source's signature (the Spike harness and the verifier call it with the source's
arguments). Phase 1 checks it before compiling and tells the repair exactly what to restore; the repair prompts
state the signature instead of forbidding any change to it. Standard C headers survive extract_code."""
from pathlib import Path

import pytest

from src.workflow.pipeline import expected_signature, extract_code, signature_mismatch

ROOT = Path(__file__).resolve().parents[1]
IGEMM = expected_signature((ROOT / "kernels/source/f32-igemm-minmax.c").read_text())


def test_signature_comes_from_the_source():
    assert IGEMM.startswith("void test_rvv(size_t mr, size_t nc, size_t kc, size_t ks,")
    assert IGEMM.endswith("const struct xnn_f32_minmax_params* restrict params)")


@pytest.mark.parametrize("params, ok", [
    ("size_t mr, size_t nc, size_t kc, size_t ks, const float** restrict a, const float* restrict w, "
     "float* restrict c, size_t cm_stride, size_t cn_stride, size_t a_offset, const float* zero, "
     "const struct xnn_f32_minmax_params* restrict params", True),
    ("size_t m, size_t n, size_t k, size_t s, const float** a, const float* w, float* c, size_t x, size_t y, "
     "size_t z, const float* zr, const struct xnn_f32_minmax_params p[1]", True),       # names, restrict, T x[N]
    ("size_t mr, size_t nc, size_t kc, size_t ks, const float** a, const float* w, float* c, size_t cm_stride, "
     "size_t cn_stride, const void* params", False),                                   # AE VLEN 128: 2 dropped
    ("size_t mr, size_t nc, size_t kc, size_t ks, const float** a, const float* w, float* c, size_t cm_stride, "
     "size_t cn_stride, size_t a_offset, const float* zero, const union xnn_f32_minmax_params params[1]", False),
])
def test_signature_mismatch(params, ok):
    assert (signature_mismatch(f"void test_rvv({params}) {{ }}", IGEMM) == "") == ok


def test_every_shipped_target_matches_its_source():
    for t in sorted((ROOT / "kernels/target").glob("*.c")):
        sig = expected_signature((ROOT / "kernels/source" / t.name).read_text())
        assert signature_mismatch(t.read_text(), sig) == "", t.name


def test_extract_code_keeps_standard_headers_only():
    code = extract_code('```c\n#include <math.h>\n#include <riscv_vector.h>\n#include "src/xnnpack/common.h"\n'
                        'void test_rvv(void) {}\n```')
    assert "#include <math.h>" in code and "riscv_vector" not in code and "xnnpack" not in code
