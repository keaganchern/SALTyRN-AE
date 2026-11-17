"""C-level tests on float values. A symbolic f32 clamp parameter is read as a symbolic scalar (salt_param), so a
C `if` on it is decided by the parameter constraints, or refused when they leave it open; read natively it was a
NaN-payload sentinel and C took the NaN branch, so a kernel wrong only for real bounds could verify. A C-level test
on a float the verifier cannot track (input or weight data read natively) is refused by the harness generator."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.workflow.verification.engine.emit import _c_float_tests

ROOT = Path(__file__).resolve().parents[1]
BASE = """static inline int is_neg(float x) { return x < 0.0f; }
static inline int unused(float y) { return y != y; }
void test_rvv(size_t mr, const float* restrict a, const float* restrict w, float* restrict c,
              const struct xnn_f32_minmax_params* restrict params) {
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  if (vmax != vmax) return;
  %s
}"""


@pytest.mark.parametrize("code, flagged", [
    ("const float a0v = *a; if (a0v != a0v) return;", True),         # NaN check on input data
    ("if (is_neg(*w)) return;", True),                               # a helper's test, called with weight data
    ("int k = (int) w[0]; if (k > 0) return;", True),                # through an int conversion
    ("float s = __riscv_vfmv_f_s_f32m1_f32(v); if (s > 0.0f) return;", True),   # a lane read out of a vector
    ("if (a[3] >= vmin) return;", True),
    ("c[0] = *a < vmin ? vmin : *a;", True),
    ("if (isnan(w[1])) return;", True),
    ("if (is_neg(vmin)) return;", False),                            # the clamp through a helper: symbolic
    ("const float r = vmax - vmin; if (r > 0.0f) return;", False),
])
def test_guard_flags_only_untrackable_float_tests(code, flagged):
    assert bool(_c_float_tests(BASE % code)) == flagged


@pytest.mark.parametrize("folder", ["kernels/source", "kernels/target", "kernels/phase1/vlen128", "kernels/phase1/vlen512",
                                    "kernels/phase2/vlen128", "kernels/phase2/vlen512"])
def test_guard_passes_the_shipped_kernels(folder):
    files = sorted((ROOT / folder).glob("*.c"))
    if not files:
        pytest.skip(f"{folder} not checked out")
    assert {f.name: h for f in files if (h := _c_float_tests(f.read_text()))} == {}


GEMM = (Path(__file__).parent / "data/f32-gemm-minmax-mu.c").read_text()     # VLEN-128 translation, no C tests
CLAMP = [ln for ln in GEMM.splitlines() if "_vf_f32m2_mu(" in ln]


def wrapped(cond, then, other):
    first, last = GEMM.index(CLAMP[0]), GEMM.index(CLAMP[-1]) + len(CLAMP[-1])
    return GEMM[:first] + f"if ({cond}) {{\n{then}\n    }} else {{\n{other}\n    }}" + GEMM[last:]


def sweep(tmp_path, src):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/f32-gemm-minmax.c").write_text(src)
    subprocess.run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", "128",
                    "--seconds", "200", "--max-level", "1", "--probe-share", "0", "--rvv-dir", str(tmp_path / "rvv"),
                    "--kernel", "f32-gemm-minmax", "--output", str(tmp_path / "out")],
                   cwd=ROOT, capture_output=True, text=True, timeout=900, check=True)
    return json.loads((tmp_path / "out/summary.json").read_text())[0]


def test_wrong_real_bound_branch_is_caught(tmp_path):
    # Correct only on the NaN-bound branch, which the "neither NaN" constraint rules out.
    lower = "\n".join(ln for ln in CLAMP if "vfmax_vf" in ln)
    r = sweep(tmp_path, wrapped("vmax != vmax", "\n".join(CLAMP), lower))
    assert r["status"] == "COUNTEREXAMPLE", r


def test_branch_the_constraints_decide_verifies(tmp_path):
    garbage = "      vacc0 = __riscv_vfmerge_vfm_f32m2(vacc0, vmax, mask0, vl);"
    r = sweep(tmp_path, wrapped("vmin <= vmax", "\n".join(CLAMP), garbage))
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 1, r.get("failure", r)


def test_branch_the_constraints_leave_open_forks(tmp_path):
    # The constraints leave `vmax > 0` open: the shape is proved once per outcome (each path asserting its own), and
    # verifies only if both do. Both branches clamp correctly here.
    r = sweep(tmp_path, wrapped("vmax > 0.0f", "\n".join(CLAMP), "\n".join(CLAMP)))
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 1, r.get("failure", r)
    shapes = [json.loads(l) for l in (tmp_path / "out/f32-gemm-minmax/shapes.jsonl").read_text().splitlines()]
    assert any(e.get("status") == "PATHS" and e.get("paths") == 2 for s in shapes for e in s.get("events", [])), shapes[:1]


def test_branch_the_constraints_leave_open_is_checked_on_both_paths(tmp_path):
    # Wrong on one side of the open branch only: that path's proof finds it.
    lower = "\n".join(ln for ln in CLAMP if "vfmax_vf" in ln)
    r = sweep(tmp_path, wrapped("vmax > 0.0f", "\n".join(CLAMP), lower))
    assert r["status"] == "COUNTEREXAMPLE", r


def test_rvv_reference_bound_tests_are_symbolic():
    # An RVV (Phase-1) reference that tests its clamp bounds in C reads them symbolically, like a candidate: through
    # the NaN-payload sentinel it took its NaN-bound branch, and Phase 2 failed against a correct Phase 1.
    from src.workflow.verification.engine.emit import _tests_clamp_in_c
    assert _tests_clamp_in_c(BASE % "")
    assert not _tests_clamp_in_c(GEMM)
    shipped = [f.relative_to(ROOT).as_posix() for d in ("kernels/phase1/vlen128", "kernels/phase1/vlen512")
               for f in sorted((ROOT / d).glob("*.c")) if _tests_clamp_in_c(f.read_text())]
    assert shipped == ["kernels/phase1/vlen512/f32-gemm-minmax.c"]
