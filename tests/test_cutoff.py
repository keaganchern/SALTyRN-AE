from pathlib import Path

import pytest

from src.workflow.verification.engine.cutoff import cutoff, strip_mined

ROOT = Path(__file__).resolve().parents[1]
HEAD = "void test_rvv(size_t n, const float* x, float* y) {\n"


def kernel(body):
    return HEAD + body + "\n}\n"


def test_plain_strip_mined_loop_cuts_off_at_vlmax():
    src = kernel("""  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t v = __riscv_vle32_v_f32m8(x, vl);
    __riscv_vse32_v_f32m8(y, v, vl);
    x += vl; y += vl; n -= vl;
  }""")
    assert strip_mined(src).ok
    assert cutoff(src, 512) == 128 and cutoff(src, 128) == 32


def test_carried_accumulator_is_rejected():
    src = kernel("""  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    acc = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(x, vl), acc, vl);
    x += vl; n -= vl;
  }""")
    assert not strip_mined(src).ok and "acc" in strip_mined(src).reason


def test_fixed_stride_main_loop_then_tail():
    src = kernel("""  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  for (; n >= 2 * vlmax; n -= 2 * vlmax) {
    vfloat32m8_t a = __riscv_vle32_v_f32m8(x, vlmax);
    vfloat32m8_t b = __riscv_vle32_v_f32m8(x + vlmax, vlmax);
    __riscv_vse32_v_f32m8(y, a, vlmax);
    __riscv_vse32_v_f32m8(y + vlmax, b, vlmax);
    x += 2 * vlmax; y += 2 * vlmax;
  }
  if (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    __riscv_vse32_v_f32m8(y, __riscv_vle32_v_f32m8(x, vl), vl);
  }""")
    assert strip_mined(src).ok and cutoff(src, 512) == 256


def test_peeled_first_chunk_is_rejected():
    src = kernel("""  size_t vl0 = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t v = __riscv_vle32_v_f32m8(x, vl0);
  x += vl0; n -= vl0;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    __riscv_vse32_v_f32m8(y, v, vl);
    v = __riscv_vle32_v_f32m8(x, vl);
    x += vl; y += vl; n -= vl;
  }""")
    assert not strip_mined(src).ok


@pytest.mark.parametrize("name", ["f32-vadd", "qs8-vadd-minmax", "f32-f16-vcvt"])
def test_phase1_elementwise_targets_qualify(name):
    assert strip_mined((ROOT / "kernels/target" / f"{name}.c").read_text()).ok


@pytest.mark.parametrize("name", ["qs8-rsum", "qu8-rsum"])
def test_reductions_do_not(name):
    assert not strip_mined((ROOT / "kernels/target" / f"{name}.c").read_text()).ok


def test_chunk_locality_is_syntactic_only_for_unit_stride_accesses():
    plain = kernel("""  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    __riscv_vse32_v_f32m8(y, __riscv_vle32_v_f32m8(x, vl), vl);
    x += vl; y += vl; n -= vl;
  }""")
    assert strip_mined(plain).local
    strided = plain.replace("__riscv_vle32_v_f32m8(x, vl)", "__riscv_vlse32_v_f32m8(x, 8, vl)")
    assert strip_mined(strided).ok and not strip_mined(strided).local
    ahead = plain.replace("__riscv_vle32_v_f32m8(x, vl)", "__riscv_vle32_v_f32m8(x, vl + 1)")
    assert not strip_mined(ahead).local


def test_hidden_effects_reject_the_premise():
    macro = "#define STEP(v) acc = v\n" + kernel("""  float acc = 0;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    STEP(1.0f);
    __riscv_vse32_v_f32m8(y, __riscv_vle32_v_f32m8(x, vl), vl);
    x += vl; y += vl; n -= vl;
  }""")
    assert not strip_mined(macro).ok and "macro" in strip_mined(macro).reason
    addr = kernel("""  float acc = 0;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    bump(&acc);
    __riscv_vse32_v_f32m8(y, __riscv_vle32_v_f32m8(x, vl), vl);
    x += vl; y += vl; n -= vl;
  }""")
    assert not strip_mined(addr).ok and "address" in strip_mined(addr).reason


POOL_DERIVED = {"input_offset": "0", "input_pixel_stride": "0",
                "input_increment": "kernel_elements * sizeof(void*)", "output_increment": "channels * sizeof(float)"}


def test_pool_channels_split_lane_by_lane():
    from src.workflow.verification.engine.cutoff import lane_split
    for name, derived in (("f32-maxpool", POOL_DERIVED),
                          ("f32-argmaxpool", {**POOL_DERIVED, "input_increment": "pooling_elements * sizeof(void*)",
                                              "index_increment": "channels * sizeof(uint32_t)"})):
        ls = lane_split((ROOT / "kernels/target" / f"{name}.c").read_text(),
                        (ROOT / "kernels/source" / f"{name}.c").read_text(), "channels", derived)
        assert ls.ok and ls.base == 4, (name, ls.reason)


def test_cross_lane_ops_and_other_uses_break_the_split():
    from src.workflow.verification.engine.cutoff import lane_split
    rvv = (ROOT / "kernels/target/f32-maxpool.c").read_text()
    neon = (ROOT / "kernels/source/f32-maxpool.c").read_text()
    first = "vacc = __riscv_vfmax_vv_f32m8(vacc, __riscv_vle32_v_f32m8(i1, vl), vl);"
    assert first in rvv
    slid = rvv.replace(first, first + " vacc = __riscv_vslidedown_vx_f32m8(vacc, 1, vl);", 1)
    assert "vslidedown" in lane_split(slid, neon, "channels", POOL_DERIVED).reason
    used = rvv.replace("float* o = output;", "float* o = output + (channels & 1);", 1)
    assert "outside the channel loops" in lane_split(used, neon, "channels", POOL_DERIVED).reason
    carried = neon.replace("vst1q_f32(o, vacc); o += 4;", "vst1q_f32(o, vacc); o += 4; vlast = vacc;", 1)
    carried = carried.replace("do {", "float32x4_t vlast = vmin;\n  do {", 1)
    assert "vlast" in lane_split(rvv, carried, "channels", POOL_DERIVED).reason


def test_neon_tile_loops_are_outer_loops():
    import re
    from src.workflow.verification.engine.cutoff import outer_loop
    for name, tile in (("f32-gemm-minmax", "8"), ("f32-igemm-minmax", "8"), ("qd8-f32-qc4w-gemm-minmax", "16")):
        src = re.sub(r"\btest_neon\b", "test_rvv", (ROOT / "kernels/source" / f"{name}.c").read_text())
        ol = outer_loop(src, {})
        assert ol.ok and ol.dim == "nc" and ol.stride == tile, (name, ol.reason)


def test_dwconv_channels_split_lane_by_lane():
    from src.workflow.verification.engine.cutoff import lane_split
    ls = lane_split((ROOT / "kernels/target/f32-dwconv-minmax.c").read_text(),
                    (ROOT / "kernels/source/f32-dwconv-minmax.c").read_text(), "channels",
                    {"input_stride": "3 * sizeof(void*)", "output_increment": "0"})
    assert ls.ok and ls.base == 4


def test_width_changing_reinterpret_and_repointed_chunks_break_the_split():
    from src.workflow.verification.engine.cutoff import lane_split
    rvv = (ROOT / "kernels/target/f32-maxpool.c").read_text()
    neon = (ROOT / "kernels/source/f32-maxpool.c").read_text()
    store = "__riscv_vse32_v_f32m8(o, vacc, vl); o += vl;"
    assert store in rvv
    wide = rvv.replace(store, "vuint8m8_t vb = __riscv_vreinterpret_v_f32m8_u8m8(vacc); " + store, 1)
    assert "element width" in lane_split(wide, neon, "channels", POOL_DERIVED).reason
    repoint = rvv.replace(store, store + " o = output;", 1)
    assert "own chunk" in lane_split(repoint, neon, "channels", POOL_DERIVED).reason
