"""The RVV models where vl, VLMAX and tail elements meet (RVV spec 16.3-16.5, 3.4.3).

A slide or gather reads source elements up to VLMAX whatever vl is, and reads 0 only past VLMAX; the models stopped
at vl, so `vslidedown(v, 1, vl)` saw a 0 where the hardware reads v[vl]. A tail element (a lane past the vl of the
operation that produced a vector, the lanes of vmv.s.x and of a reduction's result past element 0, an undefined
register) is unspecified; the models made it 0, so a kernel whose outputs depend on one could be proved against a
zero. Each is now the spec's value or a fresh unconstrained one, and a plain read of a lane a vector does not have
(an out-of-range std::vector access before, a crash or a stale term) is refused (NATIVE_ACCESS)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.workflow.verification.orchestrator import compile_harness  # noqa: E402


def test_tail_and_vlmax_semantics(tmp_path):
    src = (ROOT / "tests/data/rvv_tail_semantics.cpp").read_text()
    b, err = compile_harness(src, "rvv_tail_semantics", output_dir=tmp_path, backend="cvc5")
    assert b, err[-2000:]
    r = subprocess.run([str(b)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "ALL OK" in r.stdout, r.stdout + r.stderr
