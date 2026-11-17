"""The masked-undisturbed vfmin/vfmax (__riscv_vf{min,max}_vf_f32m2_mu, rvv/ops/policy.hpp): active lanes get the
op, masked-off lanes keep maskedoff. data/f32-gemm-minmax-mu.c (a VLEN-128 translation from the AE run) uses them to
propagate NaN like Neon's vminq/vmaxq: mask = (acc == acc), so a NaN lane keeps its NaN. It must verify; a mutant that
clamps the upper bound with vfmax instead of vfmin must not. Rows alias at mr = 1 (row 0 is overwritten), so the
mutant shows from mr = 2, which level 2 includes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOOD = (Path(__file__).parent / "data/f32-gemm-minmax-mu.c").read_text()
LINE = "vacc0 = __riscv_vfmin_vf_f32m2_mu(mask0, vacc0, vacc0, vmax, vl);"


def sweep(tmp_path, src):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/f32-gemm-minmax.c").write_text(src)
    subprocess.run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", "128",
                    "--seconds", "240", "--max-level", "2", "--probe-share", "0", "--rvv-dir", str(tmp_path / "rvv"),
                    "--kernel", "f32-gemm-minmax", "--output", str(tmp_path / "out")],
                   cwd=ROOT, capture_output=True, text=True, timeout=900, check=True)
    return json.loads((tmp_path / "out/summary.json").read_text())[0]


def test_masked_minmax_translation_verifies(tmp_path):
    r = sweep(tmp_path, GOOD)
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 2, r.get("failure", r)


def test_wrong_masked_op_is_rejected(tmp_path):
    assert GOOD.count(LINE) == 1
    r = sweep(tmp_path, GOOD.replace(LINE, LINE.replace("vfmin_vf_f32m2_mu", "vfmax_vf_f32m2_mu")))
    assert r["status"] == "COUNTEREXAMPLE", r
