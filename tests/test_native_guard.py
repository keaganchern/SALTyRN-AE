"""What a kernel computes natively is invisible to the symbolic model; these are the checks that keep it from
making a proof pass.

Run time (core/native_guard.hpp, around every kernel call): a native store into a registered buffer, native float
arithmetic on a NaN-payload sentinel (signaling, so any operation quiets it), an intrinsic without a model, and an
rsum side channel used other than `*out += sum` are refused (NATIVE_ACCESS). Each was a wrong VERIFIED before:
`vfmacc_vf(acc, a + 1.0f, w)` verified as `vfmacc_vf(acc, a, w)`, and a kernel that overwrote `c[0] = 42` verified.

Static (emit.py, _native_scalar_uses with _c_float_tests): the native uses the run time cannot see (a test, an
index, a bit-cast, a library call such as fabsf, a float-to-int conversion) are refused, in macro bodies too, and a
kernel refused only for testing an input float it reads natively (a zero-skip) is verified with those reads made
symbolic instead (_symbolic_tested_reads), its tests forking into paths."""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.workflow.verification.engine.emit import (  # noqa: E402
    _c_float_tests, _native_scalar_uses, _strip_kernel_redeclarations, _symbolic_tested_reads)

HEAD = ("void test_rvv(size_t n, const float* a0, const float** id, const float* zero, float* out, int32_t* o32, "
        "const struct xnn_f32_minmax_params* params) {\n")
REFUSED = {
    "zero-skip": "const float a_val = *a0++; if (a_val != 0.0f) { acc = __riscv_vfmacc_vf_f32m1(acc, a_val, w, vl); }",
    "truthiness": "float a_val = a0[0]; if (a_val) n++;",
    "not": "float a_val = a0[0]; n += !a_val;",
    "ternary": "float a_val = a0[0]; n = a_val ? 1 : 2;",
    "macro condition": "float a_val = a0[0]; if XNN_UNLIKELY(a_val) n++;",
    "fabsf": "float a_val = a0[0]; acc = __riscv_vfmv_v_f_f32m1(fabsf(a_val), vl);",
    "memcpy bit-cast": "float a_val = a0[0]; uint32_t u; memcpy(&u, &a_val, 4);",
    "pointer bit-cast": "float a_val = a0[0]; uint32_t* q = (uint32_t*) &a_val;",
    "int cast": "float a_val = a0[0]; n = (int32_t) a_val;",
    "int cast of an expression": "n = (int32_t) (a0[0] * 2.0f);",
    "int declaration": "int32_t i = a0[0];",
    "int assignment": "int32_t i; i = a0[1];",
    "functional cast": "n = int32_t(a0[0]);",
    "index": "float a_val = a0[0]; n = out[(int) a_val];",
    "struct member": "struct xnn_f32_minmax_params p2; p2.scalar.min = a0[0];",
    "float extract tested": "float m = __riscv_vfmv_f_s_f32m1_f32(v); if (m > 0) n++;",
    "int extract handed back": "int32_t s = __riscv_vmv_x_s_i32m1_i32(v); v = __riscv_vadd_vx_i32m1(v, s, vl);",
    "int extract tested": "int32_t s = __riscv_vmv_x_s_i32m1_i32(v); if (s > 0) *o32 += s;",
    "local array": "float buf[4]; buf[0] = a0[0]; if (buf[0] > 0) n++;",
    "isnan": "if (isnan(a0[0])) n++;",
    "&& operand": "float a_val = a0[0]; if (n > 2 && a_val) n++;",
}
ACCEPTED = {
    "handed to an intrinsic": "const float a_val = *a0++; acc = __riscv_vfmacc_vf_f32m1(acc, a_val, w, vl);",
    "arithmetic (refused at run time)": "float a_val = a0[0] * 2.0f; acc = __riscv_vfmv_v_f_f32m1(a_val, vl);",
    "rsum side channel": "int32_t s = __riscv_vmv_x_s_i32m1_i32(v); *o32 += s;",
    "pointer test": "const float* i0 = id[0]; if XNN_UNPREDICTABLE(i0 != zero) { i0 = (const float*) ((uintptr_t) i0 + n); }",
    "pointer reassigned and tested": "const float* i0_base = id[0]; i0_base = id[1]; const float* i0 = (i0_base == zero) ? zero : i0_base;",
    "pointer arithmetic": "const float* p = (const float*) ((uintptr_t) id[0] + n);",
    "scalar broadcast": "acc = __riscv_vfmul_vf_f32m1(acc, *a0, vl);",
    "clamp bound": "const float vmin = params->scalar.min; acc = __riscv_vfmax_vf_f32m1(acc, vmin, vl);",
    "element address": "const float* q = &a0[n]; acc = __riscv_vle32_v_f32m1(&a0[4], vl);",
    "integer loop": "for (size_t i = 0; i < n; i++) { out[i] = 0; }",
    "copies (stores refused at run time)": "out[0] = a0[0]; *out = *a0;",
}


@pytest.mark.parametrize("body", REFUSED.values(), ids=REFUSED.keys())
def test_native_use_is_refused(body):
    assert _native_scalar_uses(HEAD + body + "\n}\n")


@pytest.mark.parametrize("body", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_native_use_is_accepted(body):
    assert not _native_scalar_uses(HEAD + body + "\n}\n")


def test_helpers_are_followed():
    test = "static inline int pos(float x) { return x > 0.0f; }\n" + HEAD + "n = pos(a0[0]);\n}\n"
    assert _native_scalar_uses(test)
    add = ("static inline vfloat32m1_t f(vfloat32m1_t v, float x, size_t vl) { return __riscv_vfadd_vf_f32m1(v, x, vl); }\n"
           + HEAD + "acc = f(acc, a0[0], vl);\n}\n")
    assert not _native_scalar_uses(add)


def test_a_test_inside_a_macro_is_seen():
    src = "#define SKIP(x) if ((x) != 0.0f)\nvoid test_rvv(const float* a0) { float a = a0[0]; SKIP(a) { a0++; } }\n"
    assert _c_float_tests(src) and _native_scalar_uses(src)


def test_alias_declarations_are_dropped():
    src = ("void test_rvv(size_t n, const float* x) { }\n"
           'void test_neon(size_t n, const float* x) __attribute__((weak, alias("test_rvv")));\n'
           'void other(size_t n, const float* x[XNN_MIN_ELEMENTS(1)])\n  __attribute__((weak, alias("test_rvv"), used));')
    assert "alias" not in _strip_kernel_redeclarations(src)


def test_zero_skip_reads_become_symbolic():
    src = HEAD + REFUSED["zero-skip"].replace("; if", ";\n  if") + "\n}\n"     # one declaration per line, as kernels are
    out = _symbolic_tested_reads(src)
    assert "const auto a_val = salt_as<float>(salt_scalar_postinc(a0));" in out
    assert not _c_float_tests(out) and not _native_scalar_uses(out)
    # a use the symbolic type has no meaning for stays refused (the source is returned unchanged)
    # fabsf on a SymbolicScalar does not compile: the rewrite is used, and the harness compile refuses the kernel
    fab = HEAD + "const float a_val = *a0++;\n  if (a_val != 0.0f) n++;\n  acc = __riscv_vfmv_v_f_f32m1(fabsf(a_val), vl);\n}\n"
    assert "fabsf(a_val)" in _symbolic_tested_reads(fab) and "salt_scalar_postinc(a0)" in _symbolic_tested_reads(fab)


def test_provided_kernels_are_untouched():
    files = sorted(glob.glob(str(ROOT / "kernels/source/*.c")) + glob.glob(str(ROOT / "kernels/phase*/vlen*/*.c")))
    assert files
    for f in files:
        src = Path(f).read_text()
        assert not _c_float_tests(src), f
        assert not _native_scalar_uses(_strip_kernel_redeclarations(src)), f
        assert _symbolic_tested_reads(src) == src, f


# ---- run time (each a proof of a few shapes: about a minute) --------------------------------------------------
GEMM = (ROOT / "kernels/phase1/vlen128/f32-gemm-minmax.c").read_text()
FMA = "vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, a0c0, vb0, vl);"
STORE = "    __riscv_vse32_v_f32m8(c0, vacc0, vl); c0 += vl;"
RSUM = (ROOT / "kernels/phase1/vlen128/qs8-rsum.c").read_text()


def gemm_sweep(tmp_path, src):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/f32-gemm-minmax.c").write_text(src)
    subprocess.run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", "128",
                    "--seconds", "300", "--max-level", "1", "--rvv-dir", str(tmp_path / "rvv"), "--kernel",
                    "f32-gemm-minmax", "--output", str(tmp_path / "out")],
                   cwd=ROOT, capture_output=True, text=True, timeout=1500, check=True)
    return json.loads((tmp_path / "out/summary.json").read_text())[0]


def rsum_sweep(tmp_path, src):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/qs8-rsum.c").write_text(src)
    env = dict(__import__("os").environ, SALT_AE_OUT=str(tmp_path / "out"))
    subprocess.run([sys.executable, str(ROOT / "artifact/elementwise.py"), "--phase", "1", "--vlen", "128",
                    "--seconds", "60", "--workers", "1", "--rvv-dir", str(tmp_path / "rvv"), "qs8-rsum"],
                   cwd=ROOT / "artifact", capture_output=True, text=True, timeout=1200, check=True, env=env)
    return json.loads(next((tmp_path / "out").glob("elementwise_phase*/summary.json")).read_text())["qs8-rsum"]


def test_gemm_control_verifies(tmp_path):
    r = gemm_sweep(tmp_path, GEMM)
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 1, r.get("failure", r)


def test_native_float_arithmetic_is_refused(tmp_path):
    assert GEMM.count(FMA) == 2
    r = gemm_sweep(tmp_path, GEMM.replace(FMA, FMA.replace("a0c0,", "a0c0 + 1.0f,")))
    assert r["status"] == "NATIVE_ACCESS", r


def test_native_store_is_refused(tmp_path):
    assert GEMM.count(STORE) == 1
    r = gemm_sweep(tmp_path, GEMM.replace(STORE, STORE.replace("c0 += vl;", "c0[0] = 42.0f; c0 += vl;")))
    assert r["status"] == "NATIVE_ACCESS", r


@pytest.mark.parametrize("store, want", [("*output += sum;", None), ("*output = sum;", "COUNTEREXAMPLE"),
                                          ("*output += sum + 1;", "NATIVE_ACCESS")])
def test_rsum_side_channel(tmp_path, store, want):
    # `*output = sum` drops the output's prior value (rsum accumulates): with the prior symbolic it now differs.
    assert RSUM.count("  *output += sum;") == 1
    r = rsum_sweep(tmp_path, RSUM.replace("  *output += sum;", "  " + store))
    if want is None:
        assert r["max_verified"] > 0, r
    else:
        assert r["max_verified"] == 0 and r["stopped"][0]["status"] == want, r


def test_rsum_folding_the_prior_value_verifies(tmp_path):
    # The shipped VLEN-128 Phase-2 qu8-rsum seeds its reduction with `*output` and stores `*output = sum`: verified
    # with the output's prior value symbolic (it used to verify only because the native read saw 0).
    src = (ROOT / "kernels/phase2/vlen128/qu8-rsum.c").read_text()
    assert "__riscv_vmv_s_x_u32m1(*output" in src
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/qu8-rsum.c").write_text(src)
    env = dict(__import__("os").environ, SALT_AE_OUT=str(tmp_path / "out"), SALT_PHASE1_DIR=str(ROOT / "kernels/phase1/vlen128"))
    subprocess.run([sys.executable, str(ROOT / "artifact/elementwise.py"), "--phase", "2", "--vlen", "128",
                    "--seconds", "60", "--workers", "1", "--rvv-dir", str(tmp_path / "rvv"), "qu8-rsum"],
                   cwd=ROOT / "artifact", capture_output=True, text=True, timeout=1200, check=True, env=env)
    r = json.loads(next((tmp_path / "out").glob("elementwise_phase*/summary.json")).read_text())["qu8-rsum"]
    assert r["max_verified"] > 0, r


# ---- what repair is told about a refusal -------------------------------------------------------------------------
from src.workflow.verification import refusal  # noqa: E402

KERNEL_SRC = "void test_rvv(const float* a0) {\n  const float a_val = *a0++;\n  if (a_val != 0.0f) {\n    x = __riscv_vfoo_v_f32m1(v, vl);\n  }\n}\n"


@pytest.mark.parametrize("reason, want", [
    ("native use of a value the verifier cannot track symbolically: test_rvv: a test on a native value: ( a_val != 0.0f )",
     "compute every case with vector intrinsics"),
    ("candidate stored natively into c_rvv at byte 2: a C store the symbolic model cannot see", "vector stores"),
    ("a native C access to a symbolic buffer's memory (a read the symbolic model cannot see)", "vector loads"),
    ("native float arithmetic on a symbolic value (its NaN-payload sentinel became 0x7fe10000)", "RVV intrinsics"),
    ("__riscv_vfoo_v_f32m1: this intrinsic has no symbolic model", "other RVV intrinsics"),
    ("the verifier has no model for __riscv_vfoo_v_f32m1", "other RVV intrinsics"),
    ("candidate used an extracted sum natively other than `*out += sum`", "*output += sum"),
    ("candidate: native use of a value the verifier cannot track symbolically: get: a native read of a param field: "
     "p [ 0 ] . scalar . slope", "not in a helper"),
    ("a read of vector element 1 of a vector produced with vl 1: a tail element, whose value RVV leaves unspecified",
     "_tu"),
])
def test_refusal_advice_fits_its_kind(reason, want):
    assert want in refusal.describe(reason, KERNEL_SRC)


def test_refusal_names_the_line():
    assert "line 3" in refusal.describe("x: test_rvv: a test on a native value: ( a_val != 0.0f )", KERNEL_SRC)
    assert "line 4" in refusal.describe("the verifier has no model for __riscv_vfoo_v_f32m1", KERNEL_SRC)
    assert "line" not in refusal.describe("candidate stored natively into c_rvv at byte 2", KERNEL_SRC)


def test_missing_intrinsics_are_parsed_from_compile_errors():
    err = ("h.cpp:12:9: error: use of undeclared identifier '__riscv_vfoo_v_f32m1'\n"
           "h.cpp:13:9: error: no matching function for call to 'salt_cvc5::__riscv_vbar_vv_i8m1'\n"
           "h.cpp:14:9: error: use of undeclared identifier 'vaddq_f32'")
    assert refusal.missing_intrinsics(err) == ["__riscv_vbar_vv_i8m1", "__riscv_vfoo_v_f32m1"]


# ---- float params tested in C ------------------------------------------------------------------------------------
# An all-float params struct is bound with NaN-payload sentinels, which intrinsics decode; a C test on a field
# (`if (slope < 1.0f)`) saw the NaN and always took the same branch. Such a kernel now reads its params as symbolic
# scalars (the test forks), and a native use the rewrite cannot reach is refused.
from src.workflow.verification.engine.emit import _native_param_reads, _native_param_uses  # noqa: E402

LRELU = "xnn_f32_lrelu_params"
PHEAD = "void test_rvv(size_t n, const float* x, float* y, const struct xnn_f32_lrelu_params* restrict params) {\n"


@pytest.mark.parametrize("body", [
    "const float slope = params->scalar.slope; if (slope < 1.0f) n++;",
    "if (params->scalar.slope != 0.0f) n++;",
    "n = (int32_t) params->scalar.slope;",
    "v = __riscv_vfmv_v_f_f32m1(fabsf(params->scalar.slope), 1);",
    "const struct xnn_f32_lrelu_params* p = params; n = p->scalar.slope > 0 ? 1 : 2;",
], ids=["local tested", "field tested", "int cast", "library call", "alias tested"])
def test_native_float_param_use_is_found(body):
    assert _native_param_uses(PHEAD + body + "\n}\n", {LRELU})


@pytest.mark.parametrize("body", [
    "const float slope = params->scalar.slope; v = __riscv_vfmul_vf_f32m1(v, slope, 1);",
    "v = __riscv_vlse32_v_f32m1(&params->scalar.slope, 0, 1);",
    "if (n > 4) n--;",
])
def test_float_param_handed_to_intrinsics_is_not(body):
    assert not _native_param_uses(PHEAD + body + "\n}\n", {LRELU})


def test_helper_reads_of_a_symbolic_params_struct_are_found():
    helper = ("static inline float get(const union xnn_f32_lrelu_params p[restrict 1]) { return p[0].scalar.slope; }\n"
              + PHEAD + "v = __riscv_vfmul_vf_f32m1(v, get(params), 1);\n}\n")
    assert _native_param_reads(helper, {LRELU})
    assert not _native_param_reads(PHEAD + "v = __riscv_vle32_v_f32m1(&params->scalar.slope, 1);\n}\n", {LRELU})


VLRELU = (ROOT / "kernels/phase1/vlen128/f32-vlrelu.c").read_text()
LOOP = VLRELU[VLRELU.index("  while (n > 0)"):VLRELU.rindex("}")]


def vlrelu_sweep(tmp_path, src):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/f32-vlrelu.c").write_text(src)
    env = dict(__import__("os").environ, SALT_AE_OUT=str(tmp_path / "out"))
    subprocess.run([sys.executable, str(ROOT / "artifact/elementwise.py"), "--phase", "1", "--vlen", "128",
                    "--seconds", "120", "--workers", "1", "--rvv-dir", str(tmp_path / "rvv"), "f32-vlrelu"],
                   cwd=ROOT / "artifact", capture_output=True, text=True, timeout=1200, check=True, env=env)
    return json.loads(next((tmp_path / "out").glob("elementwise_phase*/summary.json")).read_text())["f32-vlrelu"]


@pytest.mark.parametrize("then, want", [(LOOP, None), (LOOP.replace("vout, vl)", "vx, vl)"), "COUNTEREXAMPLE")],
                         ids=["both branches right", "slope < 1 branch wrong"])
def test_branch_on_a_float_param(tmp_path, then, want):
    # The wrong branch is the one the NaN sentinel never took: it verified before.
    assert LOOP.count("vout, vl)") == 1
    src = VLRELU.replace(LOOP, "  if (slope < 1.0f) {\n" + then + "  } else {\n" + LOOP + "  }\n")
    r = vlrelu_sweep(tmp_path, src)
    if want is None:
        assert r["max_verified"] > 0 and r["max_paths"] == 2, r
    else:
        assert r["max_verified"] == 0 and r["stopped"][0]["status"] == want, r


def test_a_tail_read_is_refused(tmp_path):
    # A Phase-2 search candidate accumulates its tail with vl < VLMAX, then adds the accumulator at VLMAX: it reads
    # tail elements RVV leaves unspecified. The model had no lanes there (an out-of-range read: the harness crashed,
    # reported as no size verified); now refused, with the reason the repair prompt gets.
    src = (ROOT / "tests/data/qs8-rsum-tail-read.c").read_text()
    r = rsum_sweep(tmp_path, src)
    assert r["max_verified"] == 0 and r["stopped"][0]["status"] == "NATIVE_ACCESS", r
    assert "tail element" in r["stderr_tail"], r
