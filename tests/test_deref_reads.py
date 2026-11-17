"""Scalar reads of integer symbolic inputs through a dereference. The AE VLEN-128 translation of
qd8-f32-qc4w-gemm-minmax reads its activations as `(int16_t) *a0++` (the paper's target uses `a0[0]`, `a0[1]`,
`a0 += 2`); the read must become a symbolic element read and the pointer must advance, while declarations,
products, stores and sizeof operands stay as they are. The same translation computes 16 columns and stores the last
tile with `__riscv_vse32_v_f32m4(c0, vout, nc)`: the store writes nc elements, not the 16 the vector holds."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.workflow.verification.engine.emit import _native_reads, _rewrite_deref_reads, _rewrite_index_reads  # noqa: E402

DATA = Path(__file__).parent / "data"
QD8 = (DATA / "qd8-f32-qc4w-gemm-minmax-postinc.c").read_text()
READS = "      const int16_t a_val0 = (int16_t) *a0++;\n      const int16_t a_val1 = (int16_t) *a0++;"


@pytest.mark.parametrize("src, want", [
    ("v = (int16_t) *a0++;", "v = salt_scalar_postinc(a0).cast<int16_t>();"),
    ("v = *a0++ * *a0;", "v = salt_scalar_postinc(a0) * salt_scalar(a0);"),
    ("if (*a0 == 0) {}", "if (salt_scalar(a0) == 0) {}"),
    ("int8_t *a0;", "int8_t *a0;"),
    ("const int8_t *a0 = a;", "const int8_t *a0 = a;"),
    ("*a0 = 3;", "*a0 = 3;"),
    ("*a0++ = 3;", "*a0++ = 3;"),
    ("y = k * a0;", "y = k * a0;"),
    ("n = sizeof(*a0);", "n = sizeof(*a0);"),
])
def test_deref_rewrite(src, want):
    assert _rewrite_deref_reads(src, "a0") == want


def test_qd8_postinc_reads_are_symbolic():
    assert QD8.count(READS) == 1
    out = _rewrite_index_reads(QD8, ["a"])
    assert out.count("salt_scalar_postinc(a0).cast<int16_t>()") == 2
    assert _native_reads(out, ["a"]) is None


def sweep(tmp_path, src, level, seconds=400):
    (tmp_path / "rvv").mkdir()
    (tmp_path / "rvv/qd8-f32-qc4w-gemm-minmax.c").write_text(src)
    subprocess.run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", "128",
                    "--seconds", str(seconds), "--max-level", str(level), "--probe-share", "0", "--rvv-dir",
                    str(tmp_path / "rvv"), "--kernel", "qd8-f32-qc4w-gemm-minmax", "--output", str(tmp_path / "out")],
                   cwd=ROOT, capture_output=True, text=True, timeout=seconds + 800, check=True)
    return json.loads((tmp_path / "out/summary.json").read_text())[0]


def test_qd8_postinc_translation_verifies(tmp_path):
    r = sweep(tmp_path, QD8, 1)
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 1, r.get("failure", r)


def test_qd8_swapped_activations_are_caught(tmp_path):
    # a_val0 and a_val1 swapped: each nibble is multiplied by the other k's activation. The counterexample goes
    # through the float dequantization (symbolic scales and bias): close to 400 s on a loaded machine.
    src = QD8.replace(READS, "      const int16_t a_val1 = (int16_t) *a0++;\n      const int16_t a_val0 = (int16_t) *a0++;")
    assert sweep(tmp_path, src, 1, seconds=1200)["status"] == "COUNTEREXAMPLE"


def test_qd8_tail_store_past_nc_is_a_memory_violation(tmp_path):
    tail = "      __riscv_vse32_v_f32m4(c0, vout, nc);"
    assert QD8.count(tail) == 1
    assert sweep(tmp_path, QD8.replace(tail, "      __riscv_vse32_v_f32m4(c0, vout, 16);"), 1)["status"] == "MEMORY"
