"""Build the perf_compare ELFs that time the provided kernels on Saturn (timing only: no output checks).

  ./ae.sh speedups prepare --vlen 512            ELFs for one VLEN (default: 128,512)

Lanes per kernel (perf_compare, third_party/autocomp/scripts/gen_perf_compare.py):
  Upstream            XNNPACK's RVV kernel (XNNPACK 3ed73f57), the baseline every speedup divides by
  Phase 1 (Sol)       kernels/phase1/vlen<V>
  Phase 2 (Autocomp)  kernels/phase2/vlen<V>
  Neon2RVV, GCC autovec: the generator's own, from the same XNNPACK
  SIMDe               the Neon kernels through SIMDe v0.8.4-rc3 (RVV mode; portable mode where RVV mode crashes)
One ELF per lane group, so a hang costs one group:
  main      Upstream, Phase 1, Phase 2       neon2rvv  Neon2RVV       autovec  GCC autovec
  simde-rvv SIMDe, RVV mode                  simde-portable (VLEN 512 only) SIMDe, portable mode, 4 kernels
  phase1    (VLEN 128 only) Upstream, Phase 1: the Phase-1 column at VLEN 128
  autovec-gemm (VLEN 512 only) GCC autovec of the 4 GEMM-family kernels

Output: $SALT_AE_OUT (default out/) /speedups/vlen<V>/: inputs/ (lane sources), app-<group>/, build-<group>/
(FireSim ELF: PERF_NO_SANITY, PERF_CSV_ONLY, 10 warmup + 100 timed iterations).

Environment: XNNPACK_SRC (XNNPACK's src/ at 3ed73f57), fetched into out/deps/ when unset; SIMDE_INCLUDE (SIMDe
v0.8.4-rc3), default third_party/simde (./ae.sh setup), else fetched into out/deps/. SALT_ZEPHYR:
zephyr-chipyard-sw (default third_party/zephyr)."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AC = REPO / "third_party/autocomp"
XNNPACK_COMMIT = "3ed73f57b5ecaa99967bb21862b0c62f47b9bf7c"
SIMDE_TAG = "v0.8.4-rc3"
FS_FLAGS = "-DPERF_NO_SANITY=1 -DPERF_CSV_ONLY=1 -DNUM_ITERATIONS=100 -DNUM_WARMUP=10"
# SIMDe in RVV mode crashes on these at VLEN 512 (GCC 14.3 spills fixed-length vector types wrongly); they run in
# portable mode instead. At VLEN 128 every kernel runs in RVV mode.
SIMDE_PORTABLE = {128: [], 512: ["f32-dwconv2d-chw", "f32-gemm-minmax", "f32-igemm-minmax", "x32-packw-gemm-goi"]}
LABEL = {"qu8-vadd-minmax": "qu8-vadd", "qs8-vadd-minmax": "qs8-vadd", "qs8-vmul-minmax-fp32": "qs8-vmul"}
AUTOVEC_OWN_ELF = {512: ["qd8-f32-qc4w-gemm-minmax", "x32-packw-gemm-goi", "f32-gemm-minmax", "f32-igemm-minmax"]}

# Weight layouts. perf_compare hands each lane the benchmark's canonical weights (e32m4 blocks for the depthwise
# convolution, output-channel tiles of 4 for conv-hwc2chw) unless the lane's file defines pack_weights(src, src_n,
# dst, dst_cap). The VLEN-128 kernels read other layouts, so a packing function is appended; it runs before the timed
# region and the kernels themselves are unchanged.
DW_TO_M8 = """
/* perf_compare canonical weights (e32m4 vl-blocks {bias[vl],k0[vl],k1[vl],k2[vl]}) -> this kernel's layout, the same
 * blocks at e32m8 (it reads vl = vsetvl_e32m8(c) channels per block). */
static void pack_weights(const void* src_v, size_t src_n, void* dst_v, size_t dst_cap) {
  const float* src = (const float*) src_v;
  float* dst = (float*) dst_v;
  (void) dst_cap;
  const size_t c = src_n / 4;
  size_t soff = 0, doff = 0, dbase = 0, dvl = __riscv_vsetvl_e32m8(c);
  for (size_t base = 0; base < c; ) {
    const size_t vl = __riscv_vsetvl_e32m4(c - base);
    for (size_t i = 0; i < vl; i++) {
      const size_t ch = base + i;
      if (ch >= dbase + dvl) { doff += 4 * dvl; dbase += dvl; dvl = __riscv_vsetvl_e32m8(c - dbase); }
      for (size_t q = 0; q < 4; q++) dst[doff + q * dvl + (ch - dbase)] = src[soff + q * vl + i];
    }
    soff += 4 * vl;
    base += vl;
  }
}
"""
HW_TO_VL = """
/* perf_compare canonical weights (output-channel tiles of 4: 28 rows {bias, 27 taps} x 4 lanes) -> this kernel's
 * layout, the same rows in tiles of vl = vsetvl_e32mLMUL(remaining output channels). */
static void pack_weights(const void* src_v, size_t src_n, void* dst_v, size_t dst_cap) {
  const float* src = (const float*) src_v;
  float* dst = (float*) dst_v;
  (void) dst_cap;
  const size_t oc = src_n / 28;
  size_t off = 0;
  for (size_t base = 0; base < oc; ) {
    const size_t vl = __riscv_vsetvl_e32mLMUL(oc - base);
    for (size_t i = 0; i < vl; i++) {
      const size_t ch = base + i;
      for (size_t r = 0; r < 28; r++) dst[off + r * vl + i] = src[(ch / 4) * 112 + r * 4 + ch % 4];
    }
    off += 28 * vl;
    base += vl;
  }
}
"""


def adapt(vlen, phase, k, s):
    if vlen != 128:
        return s
    if k == "f32-dwconv-minmax":
        if phase == 1:
            return s + DW_TO_M8
        # Its own pack_weights(src, channels, ...) converts e32m8 blocks into the e32m4 blocks it reads; the
        # canonical weights already are e32m4 blocks, so that function is renamed out of the way.
        assert s.count("void pack_weights(") == 1
        return s.replace("void pack_weights(", "void pack_weights_from_m8(")
    if k == "f32-conv-hwc2chw":
        return s + HW_TO_VL.replace("LMUL", "8" if phase == 1 else "4")
    return s


def lane_nr(vlen):
    """GEMM family: the weight tile width (NR) each lane's kernel reads. At VLEN 128 both sets tile by their own vl
    (e32m8). At VLEN 512 Phase 1 reads the canonical nr = 8 (16 for qc4w) and the Phase-2 kernels a tile of their
    vlmax, which the benchmark builds (NR 0)."""
    if vlen == 128:
        vl8 = "__riscv_vsetvlmax_e32m8()"
        return [f"--lane-nr={k}=Sol:{vl8},Autocomp:{vl8}" for k in ("f32-gemm-minmax", "f32-igemm-minmax")]
    return [f"--lane-nr={k}=Autocomp:0" for k in ("f32-gemm-minmax", "f32-igemm-minmax", "qd8-f32-qc4w-gemm-minmax")]


def groups(vlen, kernels):
    labels = [LABEL.get(k, k) for k in kernels]
    port = [k for k in SIMDE_PORTABLE[vlen] if k in kernels]
    g = {"main": (["--sections", "main", "--impls", "Upstream,Sol,Autocomp"], {}),
         "neon2rvv": (["--sections", "neon2rvv", "--impls", "Neon2RVV"], {}),
         "autovec": (["--sections", "autovec", "--impls", "Autocomp"], {}),
         "simde-rvv": (["--sections", "neon2rvv", "--impls", "Neon2RVV", "--simde",
                        "--kernels", ",".join(l for k, l in zip(kernels, labels) if k not in port)],
                       {"SIMDE_MODE": "rvv"})}
    if port:
        g["simde-portable"] = (["--sections", "neon2rvv", "--impls", "Neon2RVV", "--simde",
                                "--kernels", ",".join(LABEL.get(k, k) for k in port)], {"SIMDE_MODE": "portable"})
    if vlen == 128:
        # Phase 1 next to Upstream only, as the expected VLEN-128 Phase-1 values were measured: in the main ELF,
        # next to Phase 2, the same Phase-1 gemm and maxpool kernels ran 2.6x and 1.8x slower (placement).
        g["phase1"] = (["--sections", "main", "--impls", "Upstream,Sol"], {})
    gemm = [k for k in AUTOVEC_OWN_ELF.get(vlen, []) if k in kernels]
    if gemm:
        # The GCC-autovectorized GEMM family in its own ELF: at VLEN 512 the autovec ELF hung on FireSim at
        # qd8-f32-qc4w-gemm-minmax (batch 256) and lost the cells after it.
        g["autovec"] = (["--sections", "autovec", "--impls", "Autocomp",
                         "--kernels", ",".join(LABEL.get(k, k) for k in kernels if k not in gemm)], {})
        g["autovec-gemm"] = (["--sections", "autovec", "--impls", "Autocomp",
                              "--kernels", ",".join(LABEL.get(k, k) for k in gemm)], {})
    return g


def fetch(url, ref, dst, sub):
    """A pinned source tree in out/deps/ (a shallow fetch of one commit or tag)."""
    if not (dst / sub).exists():
        print(f"fetching {url} at {ref} into {dst}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for c in (["init", "-q"], ["remote", "add", "origin", url], ["fetch", "-q", "--depth", "1", "origin", ref],
                  ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(dst), *c], check=True)
    return dst / sub


def kernel_keys():
    """perf_compare names a kernel by its Autocomp problem id (sols/cgo-kernels/<id>_<name>.c) or its label."""
    keys = {}
    for f in sorted((AC / "sols/cgo-kernels").glob("*.c")):
        m = re.match(r"(\d+)_(.+)\.c$", f.name)
        k = (m[2] if m else f.stem).replace("dwconv2d-chw", "f32-dwconv2d-chw").replace("x32transpose", "x32-transposec")
        keys[k.replace("f32-f32-", "f32-")] = m[1] if m else k
    return keys


def zephyr_env():
    z = Path(os.environ.get("SALT_ZEPHYR", REPO / "third_party/zephyr"))
    if not (z / "tools/miniforge3/etc/profile.d/conda.sh").exists():
        sys.exit(f"no Zephyr toolchain in {z} (set SALT_ZEPHYR, see the README)")
    return z


def west(z, app, build, cflags, defs, log):
    d = " ".join(f"-D{k}={v}" for k, v in defs.items())
    cmd = (f"cd {z} && source scripts/set_envvars_sdk.sh > /dev/null 2>&1; "
           f"source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr && export PATH=\"$PATH:{REPO}/.venv/bin\" && "
           f"west build -p -b spike_riscv64 -d {build} {app} -- -DEXTRA_CFLAGS=\"{cflags}\" {d}")
    env = {k: v for k, v in os.environ.items() if k != "ZEPHYR_BASE"}
    with open(log, "w") as f:
        rc = subprocess.run(["bash", "-c", cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc == 0 and (build / "zephyr/zephyr.elf").exists()


def prepare(vlen, a):
    out = Path(os.environ.get("SALT_AE_OUT", REPO / "out")).resolve() / f"speedups/vlen{vlen}"
    deps = Path(os.environ.get("SALT_AE_OUT", REPO / "out")).resolve() / "deps"
    xnn = os.environ.get("XNNPACK_SRC") or str(fetch("https://github.com/google/XNNPACK.git", XNNPACK_COMMIT,
                                                    deps / "XNNPACK-3ed73f57", "src"))
    local = REPO / "third_party/simde"
    simde = os.environ.get("SIMDE_INCLUDE") or (str(local) if (local / "simde/arm/neon.h").exists() else None) or str(fetch("https://github.com/simd-everywhere/simde.git",
                                                         f"refs/tags/{SIMDE_TAG}", deps / f"simde-{SIMDE_TAG}", ""))
    keys = kernel_keys()
    p1, p2 = REPO / f"kernels/phase1/vlen{vlen}", REPO / f"kernels/phase2/vlen{vlen}"
    kernels = sorted(json.loads((p1 / "manifest.json").read_text())["kernels"])
    assert set(kernels) == set(keys), sorted(set(kernels) ^ set(keys))
    args, lanes = [], {}
    for phase, d, flag in ((1, p1, "--sol-file"), (2, p2, "--autocomp-file")):
        (out / f"inputs/phase{phase}").mkdir(parents=True, exist_ok=True)
        for k in kernels:
            dst = out / f"inputs/phase{phase}/{k}.c"
            dst.write_text(adapt(vlen, phase, k, (d / f"{k}.c").read_text()))
            args.append(f"{flag}={keys[k]}={dst}")
            lanes.setdefault(k, {"key": keys[k]})[f"phase{phase}"] = str(dst)
    (out / "lanes.json").write_text(json.dumps(lanes, indent=1))
    z = zephyr_env()
    built = {}
    for name, (extra, defs) in groups(vlen, kernels).items():
        app = out / f"app-{name}"
        shutil.rmtree(app, ignore_errors=True)
        shutil.copytree(AC / "benchmarks/perf_compare", app)
        r = subprocess.run([sys.executable, str(AC / "scripts/gen_perf_compare.py"), *extra, *lane_nr(vlen), *args],
                           env=dict(os.environ, PERF_COMPARE_APP=str(app), XNNPACK_SRC=xnn), capture_output=True,
                           text=True)
        (out / f"gen-{name}.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            sys.exit(f"vlen {vlen} {name}: gen_perf_compare.py failed, see {out}/gen-{name}.log")
        if "SIMDE_MODE" in defs:
            defs = dict(defs, SIMDE_INCLUDE=simde, SIMDE_VLEN=vlen)
        built[name] = defs
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        ok = dict(zip(built, ex.map(lambda n: west(z, out / f"app-{n}", out / f"build-{n}", FS_FLAGS, built[n],
                                                    out / f"build-{n}.log"), built)))
    for n in built:
        print(f"vlen {vlen} {n:15s} FireSim ELF: {'ok' if ok[n] else 'FAILED'}  {out}/build-{n}/zephyr/zephyr.elf",
              flush=True)
    failed = [n for n, v in ok.items() if not v]
    if failed:
        sys.exit(f"build failed: {failed} (logs in {out})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlen", default="128,512", help="comma-separated VLENs (default 128,512)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel builds (default 4)")
    a = ap.parse_args()
    for v in (int(x) for x in a.vlen.split(",")):
        prepare(v, a)


if __name__ == "__main__":
    main()
