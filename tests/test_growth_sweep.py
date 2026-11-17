import itertools
import os
import sys
import time

import pytest

from src.workflow.verification.engine.emit import shape_fixture
from src.workflow.verification.engine.frontend import parse_signature
from src.workflow.verification.engine.growth_sweep import MULTIDIM, PROJECT_ROOT, classify, execute_shape
from src.workflow.verification.engine.shape_plan import Axis, ShapePlan, _entry_asserts, derive_plan


def fixture(name):
    return shape_fixture(str(PROJECT_ROOT / "kernels/source" / f"{name}.c"),
                         str(PROJECT_ROOT / "kernels/target" / f"{name}.c"))


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_frontiers_cover_cube_once(dimensions):
    plan = ShapePlan([Axis(f"d{i}", i + 1, None) for i in range(dimensions)], {}, {}, [])
    seen = []
    for level in range(1, 5):
        shell = list(plan.frontier(level))
        assert len(shell) == level**dimensions - (level - 1)**dimensions
        for shape in shell:
            assert max(shape[f"d{i}"] // (i + 1) for i in range(dimensions)) == level
        seen += [tuple(shape.values()) for shape in shell]
    assert len(seen) == len(set(seen)) == 4**dimensions
    assert set(seen) == set(itertools.product(*(range(i + 1, 4 * (i + 1) + 1, i + 1) for i in range(dimensions))))


def test_bounded_axes_stop_and_finite_domains_finish():
    plan = ShapePlan([Axis("mr", 1, 2), Axis("nc", 1, 3)], {"pad": [0, 8]}, {}, [])
    seen = [s for level in range(1, 6) for s in plan.frontier(level)]
    assert len(seen) == 2 * 3 * 2 and not list(plan.frontier(4))
    assert plan.exhausted(4) and not plan.exhausted(3)


@pytest.mark.parametrize("name", MULTIDIM)
def test_every_integer_argument_is_planned(name):
    _, _, spec, plan = fixture(name)
    sig = parse_signature(str(PROJECT_ROOT / "kernels/source" / f"{name}.c"), "test_neon")
    ints = {p.name for p in sig.params if p.is_int_scalar}
    covered = {a.name for a in plan.axes} | set(plan.variants) | set(plan.derived)
    assert ints <= covered, ints - covered
    assert all(a.step >= 1 for a in plan.axes)


def test_sizes_units_and_bounds_come_from_the_asserts():
    _, _, _, gemm = fixture("f32-gemm-minmax")
    assert [(a.name, a.step, a.hi) for a in gemm.axes] == [("mr", 1, 4), ("nc", 1, None), ("kc", 4, None)]
    _, _, _, igemm = fixture("f32-igemm-minmax")
    assert ("ks", 8, None) in [(a.name, a.step, a.hi) for a in igemm.axes]
    assert igemm.variants["a_offset"] == [0, 4] and igemm.variants["zero_mode"] == [0, 1]
    _, _, _, pack = fixture("x32-packw-gemm-goi")
    assert pack.derived["nr"] == "2" and pack.variants["extra_bytes"] == [0, 8]
    assert {s["extra_bytes"] for s in pack.frontier(1)} == {0, 8}
    assert {s["bias_null"] for s in pack.frontier(1)} == {0, 1}      # the bias == NULL path
    _, _, _, chw = fixture("f32-dwconv2d-chw")
    assert ("input_width", 4, None) in [(a.name, a.step, a.hi) for a in chw.axes]


def test_only_entry_asserts_are_preconditions():
    src = """void test_neon(size_t channels) {
      assert(channels != 0);
      for (; channels >= 32; channels -= 32) {}
      assert(channels >= 1 && channels <= 31);
    }"""
    assert _entry_asserts(src, ["channels"]) == ["channels != 0"]


def test_candidate_only_asserts_are_checked_not_trusted():
    sig = parse_signature(str(PROJECT_ROOT / "kernels/source/f32-gemm-minmax.c"), "test_neon")
    ref = (PROJECT_ROOT / "kernels/source/f32-gemm-minmax.c").read_text()
    cand = ref.replace("assert(mr <= 4);", "assert(mr <= 4);\n  assert(nc <= 8);")
    plan = derive_plan(sig, [ref, cand], {"a_stride": "kc", "cm_stride": "nc * sizeof(float)",
                                          "cn_stride": "8 * sizeof(float)"}, {})
    assert "nc <= 8" not in plan.preconditions and plan.candidate_checks == ["nc <= 8"]
    assert [a.hi for a in plan.axes if a.name == "nc"] == [None]


@pytest.mark.parametrize("code,events,status", [
    (0, [{"status": "SOLVING"}, {"status": "VERIFIED"}], "VERIFIED"),
    (0, [{"status": "NONVACUOUS"}, {"status": "SOLVING"}, {"status": "VERIFIED"}], "VERIFIED"),
    (0, [{"status": "ALL_VERIFIED"}], "ERROR"),
    (0, [{"status": "TIMEOUT"}, {"status": "ALL_VERIFIED"}], "ERROR"),
    (-9, [{"status": "SOLVING"}, {"status": "VERIFIED"}], "ERROR"),
    (2, [{"status": "SOLVING"}, {"status": "TIMEOUT"}], "TIMEOUT"),
    (1, [{"status": "SOLVING"}, {"status": "COUNTEREXAMPLE"}], "COUNTEREXAMPLE"),
    (1, [{"status": "MEMORY"}], "MEMORY"),
    (1, [{"status": "VACUOUS_CONSTANT_OUTPUT"}], "VACUOUS_CONSTANT_OUTPUT"),
    (1, [{"status": "CANDIDATE_REJECTS"}], "CANDIDATE_REJECTS"),
    (3, [{"status": "PRECONDITION"}], "ERROR"),
])
def test_no_false_pass_from_missing_or_failed_result(code, events, status):
    assert classify(code, events) == status


def test_hard_deadline_works_without_stdout():
    start = time.monotonic()
    result = execute_shape([sys.executable, "-c", "import time; time.sleep(30)"], 0.1, os.environ.copy())
    assert result["status"] == "TIMEOUT"
    assert result["hard_deadline"]
    assert time.monotonic() - start < 2


def test_fixtures_are_nonvacuous_and_support_multiple_tiles():
    _, _, dw, _ = fixture("f32-dwconv-minmax")
    assert dw.params[1]["scalar.min"] != dw.params[1]["scalar.max"]
    assert dw.dual_weights["weights"]["rvv_packed"]          # the RVV target steps w += vl
    assert dw.dual_weights["weights"]["neon_tile"] == "4"
    assert dw.strides["input_stride"] == "3 * sizeof(void*)"
    _, _, ig, _ = fixture("f32-igemm-minmax")
    assert ig.strides["cn_stride"] == "8 * sizeof(float)"
    _, _, mp, _ = fixture("f32-maxpool")
    assert mp.strides["input_increment"] == "kernel_elements * sizeof(void*)"


@pytest.mark.parametrize("name,dim,cap", [("f32-maxpool", "output_pixels", 2), ("x32-packw-gemm-goi", "g", 2),
                                          ("f32-igemm-minmax", "nc", 16), ("qd8-f32-qc4w-gemm-minmax", "nc", 32),
                                          ("f32-gemm-minmax", "nc", 16)])
def test_outer_loop_induction_caps_the_axis(name, dim, cap):
    from src.workflow.verification.engine.growth_sweep import induction
    _, _, _, plan = fixture(name)
    ind = induction(plan, (PROJECT_ROOT / "kernels/target" / f"{name}.c").read_text(), 512)
    assert ind["applies"] and ind["dim"] == dim and ind["cap"] == cap
    assert next(a for a in plan.axes if a.name == dim).hi == cap


@pytest.mark.parametrize("name,why", [("x32-transposec", "derived stride"), ("qs8-rdsum", "derived stride")])
def test_induction_needs_strides_independent_of_the_dimension(name, why):
    from src.workflow.verification.engine.growth_sweep import induction
    _, _, _, plan = fixture(name)
    ind = induction(plan, (PROJECT_ROOT / "kernels/target" / f"{name}.c").read_text(), 512)
    assert not ind["applies"] and why in ind["reason"]


def test_threshold_probes_reach_paths_beyond_the_box():
    from src.workflow.verification.engine.shape_plan import probe_shapes, thresholds
    _, _, _, plan = fixture("qs8-rdsum")
    srcs = [(PROJECT_ROOT / "kernels/source/qs8-rdsum.c").read_text(),
            (PROJECT_ROOT / "kernels/target/qs8-rdsum.c").read_text()]
    assert 252 in thresholds(plan, srcs)["rows"]          # the int16 accumulator flush
    rows = {s["rows"] for s in probe_shapes(plan, srcs)}
    assert {252, 253, 504, 505} <= rows


@pytest.mark.parametrize("name,axis,wanted", [
    ("f32-maxpool", "kernel_elements", {14, 18}),       # `4 < k`, `8 < k` with k = kernel_elements - 9
    ("f32-argmaxpool", "pooling_elements", {9}),        # `8 < k` with k = pooling_elements
    ("qd8-f32-qc4w-gemm-minmax", "nc", {8}),            # `nc & 8`
    ("f32-dwconv2d-chw", "input_width", {33}),          # `w > vl`, w = input_width / 4, vl <= VLMAX(e32m2) = 32
])
def test_thresholds_read_offsets_bit_tests_and_vl(name, axis, wanted):
    from src.workflow.verification.engine.shape_plan import thresholds
    _, _, _, plan = fixture(name)
    srcs = [(PROJECT_ROOT / "kernels/source" / f"{name}.c").read_text(),
            (PROJECT_ROOT / "kernels/target" / f"{name}.c").read_text()]
    assert wanted <= set(thresholds(plan, srcs, 512).get(axis, []))
