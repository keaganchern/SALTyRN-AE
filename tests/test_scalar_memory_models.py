"""Kernels that move data in plain C. Two AE translations (VLEN 512):

- x32-packw-gemm-goi copies bias with `packed_weights[0] = bias[0];` and zero-fills with `= 0;`. Integer inputs have
  no NaN-payload sentinel, so the reads become salt_scalar and the stores into the output salt_deref.
- f32-conv-hwc2chw stores vectors into local arrays (`float out0_p0[4]; __riscv_vse32_v_f32m1(out0_p0, ...)`) and
  copies elements to the output in C. The arrays become scoped symbolic buffers (SaltScratch), their element reads
  symbolic, the output stores symbolic. Its channel pointers alias at one output channel (o0c1 = o0c0 ...), so a
  wrong store into channel 0 is overwritten there and shows from two channels (level 2)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"


def sweep(tmp_path, kernel, src, level):
    (tmp_path / "rvv").mkdir()
    (tmp_path / f"rvv/{kernel}.c").write_text(src)
    subprocess.run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", "512",
                    "--seconds", "400", "--max-level", str(level), "--probe-share", "0", "--rvv-dir",
                    str(tmp_path / "rvv"), "--kernel", kernel, "--output", str(tmp_path / "out")],
                   cwd=ROOT, capture_output=True, text=True, timeout=1200, check=True)
    return json.loads((tmp_path / "out/summary.json").read_text())[0]


PACKW = (DATA / "x32-packw-gemm-goi-c-stores.c").read_text()
BIAS = "        packed_weights[0] = bias[0];\n        packed_weights[1] = bias[1];"


def test_packw_c_copies_verify(tmp_path):
    r = sweep(tmp_path, "x32-packw-gemm-goi", PACKW, 2)
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 2, r.get("failure", r)


def test_packw_swapped_bias_is_caught(tmp_path):
    assert PACKW.count(BIAS) == 1
    src = PACKW.replace(BIAS, "        packed_weights[0] = bias[1];\n        packed_weights[1] = bias[0];")
    assert sweep(tmp_path, "x32-packw-gemm-goi", src, 2)["status"] == "COUNTEREXAMPLE"


CONV = (DATA / "f32-conv-hwc2chw-scratch.c").read_text()
LINE = "o0c0[0] = out0_p0[0]; o0c0 += 1;"
DECL = "          float out0_p0[4];\n"


def test_conv_scratch_arrays_verify(tmp_path):
    r = sweep(tmp_path, "f32-conv-hwc2chw", CONV, 1)
    assert r["status"] == "MAX_LEVEL_REACHED" and r["completed_level"] == 1, r.get("failure", r)


def test_conv_wrong_scratch_lane_is_caught(tmp_path):
    assert CONV.count(LINE) == 1
    r = sweep(tmp_path, "f32-conv-hwc2chw", CONV.replace(LINE, "o0c0[0] = out0_p0[1]; o0c0 += 1;"), 2)
    assert r["status"] == "COUNTEREXAMPLE", r


def test_conv_scratch_overrun_is_a_memory_violation(tmp_path):
    assert CONV.count(DECL) == 1
    r = sweep(tmp_path, "f32-conv-hwc2chw", CONV.replace(DECL, "          float out0_p0[2];\n"), 1)
    assert r["status"] == "MEMORY", r
