"""Memory footprints, non-vacuity and UF-first, end to end on compiled harnesses."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.workflow.verification.engine.emit import generate_harness, generate_shape_harness
from src.workflow.verification.orchestrator import compile_harness

ROOT = Path(__file__).resolve().parents[1]
SRC, TGT = ROOT / "kernels/source", ROOT / "kernels/target"


def build(source, label, tmp_path):
    binary, err = compile_harness(source, label, output_dir=tmp_path / label, backend="cvc5")
    assert binary, err[-1500:]
    return binary


def run(binary, args, **env):
    p = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, **env))
    return p.returncode, [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]


def shape(name, tmp_path, rvv=None, mutate=None, **kw):
    src, spec, _ = generate_shape_harness(str(SRC / f"{name}.c"), str(rvv or TGT / f"{name}.c"), name, **kw)
    return build(mutate(src) if mutate else src, name.replace("-", "_"), tmp_path), list(spec.dims)


def test_constant_reference_output_is_vacuous(tmp_path):
    clamp = "params.scalar.min = -500.0f;\n    params.scalar.max = 500.0f;"
    b, _ = shape("f32-dwconv-minmax", tmp_path, mutate=lambda s: s.replace(clamp, ""),
                 input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 120, 1, 1, 0, 0], SALT_NONVACUITY="1")   # channels, output_width, zero_mode, input_offset
    assert rc == 1 and ev[-1]["status"] == "VACUOUS_CONSTANT_OUTPUT"


def test_fixed_fixture_is_nonvacuous_and_verifies(tmp_path):
    b, _ = shape("f32-dwconv-minmax", tmp_path, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 2, 1, 1, 16], SALT_NONVACUITY="1", SALT_MEMCHECK="strict")
    assert rc == 0 and [e["status"] for e in ev] == ["NONVACUOUS", "SOLVING", "VERIFIED"]


def test_candidate_over_read_is_a_memory_failure(tmp_path):
    rvv = TGT / "f32-vadd.c"
    text = rvv.read_text()
    over = tmp_path / "f32-vadd-overread.c"
    # Load a full VLMAX vector even in the tail: correct values, out-of-bounds bytes.
    over.write_text(text.replace("__riscv_vle32_v_f32m8(input_a, vl)",
                                 "__riscv_vle32_v_f32m8(input_a, __riscv_vsetvlmax_e32m8())"))
    assert over.read_text() != text, "f32-vadd target changed; update the mutation"
    src = generate_harness(str(SRC / "f32-vadd.c"), str(over), "f32-vadd", params_mode="concrete")
    b = build(src, "vadd_overread", tmp_path)
    rc, ev = run(b, [1, 1, 512, 120], SALT_MEMCHECK="strict")
    assert rc == 1 and any(e["status"] == "MEMORY" for e in ev)


def test_uf_first_proves_same_order_fp_and_reports_it(tmp_path):
    b, dims = shape("f32-gemm-minmax", tmp_path, input_range=(-1000.0, 1000.0))
    assert dims == ["mr", "nc", "kc"]
    rc, ev = run(b, [512, 300, 2, 9, 16], SALT_UF_FIRST="mul,fma,add,sub")   # crosses the 8-column tile
    # The per-output split runs first now; its per-output UF stage proves the same.
    assert rc == 0 and ev[-1]["status"] == "VERIFIED" and ev[-1]["method"] in ("uf", "split")


def test_uf_first_still_finds_a_wrong_candidate(tmp_path):
    rvv = TGT / "f32-gemm-minmax.c"
    text = rvv.read_text()
    bad = tmp_path / "gemm-bad.c"
    bad.write_text(text.replace("w += 8;", "w += 7;", 1))
    assert bad.read_text() != text, "gemm target changed; update the mutation"
    b, _ = shape("f32-gemm-minmax", tmp_path, rvv=bad, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 1, 1, 4], SALT_UF_FIRST="mul,fma,add,sub")
    assert rc == 1 and ev[-1]["status"] == "COUNTEREXAMPLE"


def test_interval_simplification_proves_nan_guarded_max_chains(tmp_path):
    # NEON vmaxq propagates NaN through an ite; on a finite band the guards fold.
    b, _ = shape("f32-maxpool", tmp_path, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 1, 2, 2], SALT_MEMCHECK="strict", SALT_NO_LATTICE="1")
    assert rc == 0 and ev[-1]["status"] == "VERIFIED" and ev[-1]["method"] in ("simp", "split")


def test_lattice_form_proves_tree_vs_chain_and_keeps_a_dropped_tap(tmp_path):
    # NEON reduces the 9 taps as a tree, RVV as a chain; one normal form for both.
    b, _ = shape("f32-maxpool", tmp_path, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 1, 9, 4], SALT_MEMCHECK="strict")
    assert rc == 0 and ev[-1]["status"] == "VERIFIED" and ev[-1]["method"] in ("lattice", "split")
    text = (TGT / "f32-maxpool.c").read_text()
    tap = "\n      vacc = __riscv_vfmax_vv_f32m8(vacc, __riscv_vle32_v_f32m8(i8, vl), vl); i8 += vl;\n"
    assert text.count(tap) == 1, "maxpool target changed; update the mutation"
    bad = tmp_path / "maxpool-bad.c"
    bad.write_text(text.replace(tap, "\n      i8 += vl;\n"))
    b, _ = shape("f32-maxpool", tmp_path / "bad", rvv=bad, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 1, 9, 4], SALT_MEMCHECK="strict")
    assert rc == 1 and ev[-1]["status"] == "COUNTEREXAMPLE"


def test_bit_select_matches_vmerge_so_argmax_chains_coincide(tmp_path):
    # NEON vbslq(vcgtq(..)) and RVV vmerge(vmfgt(..)) are one ite chain after the rewrite.
    b, _ = shape("f32-argmaxpool", tmp_path, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 120, 1, 9, 4], SALT_MEMCHECK="strict")
    assert rc == 0 and ev[-1]["status"] == "VERIFIED"
    text = (TGT / "f32-argmaxpool.c").read_text()
    cmp = "vbool4_t vm3 = __riscv_vmfgt_vv_f32m8_b4(vi3, vmax, vl);"
    assert text.count(cmp) == 1, "argmaxpool target changed; update the mutation"
    bad = tmp_path / "argmaxpool-bad.c"
    bad.write_text(text.replace(cmp, "vbool4_t vm3 = __riscv_vmfgt_vv_f32m8_b4(vmax, vi3, vl);"))
    b, _ = shape("f32-argmaxpool", tmp_path / "bad", rvv=bad, input_range=(-1000.0, 1000.0))
    rc, ev = run(b, [512, 300, 1, 4, 1], SALT_MEMCHECK="strict")
    assert rc == 1 and ev[-1]["status"] == "COUNTEREXAMPLE"


def test_integer_normal_form_regroups_sums_and_keeps_overflow(tmp_path):
    # Flushing the int16 partial sums every 245 rows instead of 252 is the same sum
    # (no int16 overflow either way); every 259 rows overflows int16. Both are
    # multiples of the 7-row batch: a flush mid-batch leaves rows pointing at zero
    # (a 240-row flush is a real bug the verifier finds).
    text = (TGT / "qs8-rdsum.c").read_text()
    a, b = "r < 252 ? r : 252", "r > 252 ? r - 252 : 0"
    assert a in text and b in text, "qs8-rdsum target changed; update the mutation"
    for n, rows, status, method in ((245, 253, "VERIFIED", "int"), (259, 259, "COUNTEREXAMPLE", None)):
        rvv = tmp_path / f"rdsum{n}.c"
        rvv.write_text(text.replace(a, f"r < {n} ? r : {n}").replace(b, f"r > {n} ? r - {n} : 0"))
        bin_, _ = shape("qs8-rdsum", tmp_path / str(n), rvv=rvv)
        rc, ev = run(bin_, [512, 300, rows, 3], SALT_MEMCHECK="strict")
        assert ev[-1]["status"] == status and (method is None or ev[-1]["method"] in (method, "split"))


def test_xnnpack_nan_policy_splits_outputs_on_the_full_domain(tmp_path):
    # Full FP domain: NEON's NaN-propagating clamp and RVV's vfmax differ only where
    # the reference is NaN, which XNNPACK's oracle accepts. Per output, "reference not
    # NaN" makes the taps NaN-free, so the max-tree normal form applies.
    b, _ = shape("f32-maxpool", tmp_path)
    rc, ev = run(b, [512, 300, 1, 9, 4], SALT_MEMCHECK="strict", SALT_NAN_POLICY="both")
    assert ev[-1]["status"] == "COUNTEREXAMPLE"
    rc, ev = run(b, [512, 300, 1, 9, 4], SALT_MEMCHECK="strict", SALT_NAN_POLICY="ref")
    assert rc == 0 and ev[-1]["status"] == "VERIFIED" and ev[-1]["method"] == "split"
    text = (TGT / "f32-maxpool.c").read_text()
    tap = "\n      vacc = __riscv_vfmax_vv_f32m8(vacc, __riscv_vle32_v_f32m8(i8, vl), vl); i8 += vl;\n"
    bad = tmp_path / "maxpool-bad.c"
    bad.write_text(text.replace(tap, "\n      i8 += vl;\n"))
    b, _ = shape("f32-maxpool", tmp_path / "bad", rvv=bad)
    # One channel: the dropped ninth tap shows at any channel count, and the solver finds it in about a minute
    # (at four channels it took longer than the 300 s budget on an AWS c5 core).
    rc, ev = run(b, [512, 300, 1, 9, 1], SALT_MEMCHECK="strict", SALT_NAN_POLICY="ref")
    assert rc == 1 and ev[-1]["status"] == "COUNTEREXAMPLE"
