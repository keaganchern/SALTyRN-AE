"""UF-first bounds an abstracted product by its operands' ranges, seeded from the harness's parameter
constraints. Those constraints bound the int32 parameter fields, but a kernel may read a narrower view: qs8-vlrelu's
(int16_t) params->scalar.input_zero_point is the low half, a different term. The low bits of a value whose range
fits them have that range (bv_seeds), so the shifted input stays within +-32640 and the product within
+-32640*32768 < 2^30. Then the doubling form of vqrdmulh (vwmul, vsll 1, vnclip 16), which differs from Neon only
when 2ab overflows (a = b = -32768), verifies in the UF stage; without the low-bits range the stage found a
spurious counterexample and the exact query timed out (a VLEN-128 translation from the AE run). Kernels that are
wrong only at the edge of the zero point's range must still fail."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.workflow.verification.engine.emit import generate_harness
from src.workflow.verification.orchestrator import compile_harness

ROOT = Path(__file__).resolve().parents[1]
DECL = "const int16_t input_zero_point = (int16_t) params->scalar.input_zero_point;"
DOUBLING = r'''void test_rvv(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t input_zero_point = (int16_t) params->scalar.input_zero_point;
  const int16_t positive_multiplier = (int16_t) (-params->scalar.positive_multiplier);
  const int16_t negative_multiplier = (int16_t) (-params->scalar.negative_multiplier);
  const int16_t output_zero_point = (int16_t) params->scalar.output_zero_point;

  while (batch > 0) {
    const size_t vl = __riscv_vsetvl_e8m2(batch);

    const vint8m2_t vx = __riscv_vle8_v_i8m2(input, vl);
    vint16m4_t vacc = __riscv_vsext_vf2_i16m4(vx, vl);
    vacc = __riscv_vrsub_vx_i16m4(vacc, input_zero_point, vl);

    const vbool4_t vmask = __riscv_vmslt_vx_i16m4_b4(vacc, 0, vl);
    vacc = __riscv_vsll_vx_i16m4(vacc, 7, vl);

    vint16m4_t vmultiplier = __riscv_vmv_v_x_i16m4(negative_multiplier, vl);
    vmultiplier = __riscv_vmerge_vxm_i16m4(vmultiplier, positive_multiplier, vmask, vl);

    vint32m8_t vprod = __riscv_vwmul_vv_i32m8(vacc, vmultiplier, vl);
    vprod = __riscv_vsll_vx_i32m8(vprod, 1, vl);
    vacc = __riscv_vnclip_wx_i16m4(vprod, 16, __RISCV_VXRM_RNU, vl);

    vacc = __riscv_vsadd_vx_i16m4(vacc, output_zero_point, vl);
    const vint8m2_t vy = __riscv_vnclip_wx_i8m2(vacc, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m2(output, vy, vl);

    input += vl;
    output += vl;
    batch -= vl;
  }
}'''


def events(tmp_path, src, seconds=60):
    rvv = tmp_path / "qs8-vlrelu.c"
    rvv.write_text(src)
    h = generate_harness(str(ROOT / "kernels/source/qs8-vlrelu.c"), str(rvv), "qs8-vlrelu",
                         params_mode="symbolic", input_range=None)
    b, err = compile_harness(h, "qs8-vlrelu", output_dir=tmp_path / "build", backend="cvc5")
    assert b, err[-2000:]
    env = dict(os.environ, SALT_MEMCHECK="strict", SALT_UF_FIRST="mul,fma,add,sub,bvmul")   # the sweep's settings
    env.pop("SALT_NAN_POLICY", None)
    p = subprocess.run([str(b), "1", "1", "128", str(seconds)], capture_output=True, text=True, env=env,
                       timeout=seconds + 120)
    return [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]


def test_int16_view_of_a_bounded_param_bounds_the_product(tmp_path):
    assert DOUBLING.count(DECL) == 1 and "__riscv_vsll_vx_i32m8(vprod, 1, vl)" in DOUBLING
    ev = events(tmp_path, DOUBLING)
    assert any(e.get("status") == "VERIFIED" and e.get("method") == "uf" for e in ev), ev


@pytest.mark.parametrize("edge, other", [(127, 126), (-128, -127)])
def test_kernel_wrong_only_at_the_range_edge_still_fails(tmp_path, edge, other):
    zp = "params->scalar.input_zero_point"
    src = DOUBLING.replace(DECL, f"const int16_t input_zero_point = (int16_t) ({zp} == {edge} ? {other} : {zp});")
    ev = events(tmp_path, src)
    assert any(e.get("status") == "COUNTEREXAMPLE" for e in ev), ev
