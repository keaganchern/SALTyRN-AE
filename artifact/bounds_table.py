"""Bounds table: per kernel, the Phase-1 and Phase-2 verification bounds, read from the summaries the sweeps write
(elementwise.py, multidim.sh, or growth_sweep directly). Runs no solver.

  python bounds_table.py                      # the VLEN-512 runs under $SALT_AE_OUT (default artifact/out)
  python bounds_table.py --vlen 128           # the VLEN-128 runs (elementwise_phase<N>_vlen128/, multidim-vlen128/)
  python bounds_table.py --phase1 P ... --phase2 P ...   # any summary.json files, or directories holding them
  python bounds_table.py ... --csv FILE       # also write the table as CSV

Default inputs, under $SALT_AE_OUT; a later input replaces an earlier one for the same kernel:
  Phase 1: elementwise_phase1/, multidim/phase1/
  Phase 2: elementwise_phase2/, multidim/phase2/
(with _vlen<V> / -vlen<V> suffixes for another VLEN; inputs that do not exist are skipped).

Elementwise bound: the largest n such that every size 1..n verified. Multidimensional bound: the region in which every
shape verified, dimensions with the same range grouped, "every" for a dimension proved for all sizes by induction;
test-harness variants (zero_mode, offsets, bias_null, extra_bytes) are left out. A kernel that stopped on a
counterexample or memory violation shows that verdict instead. Tagged elementwise runs (see elementwise.py) are
combined: @multeq/@multne as "eq (a_multiplier == b_multiplier) / ne (!=)", @neon as "(against Neon: n)"; @ref
(XNNPACK's NaN rule) is shown like an untagged run."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import paths

# The paper's table order; kernels not listed follow alphabetically.
ORDER = ["f32-vadd", "f32-vsub", "f32-vmul", "f32-vdiv", "f32-vmax", "f32-vmin", "f32-vmulc", "f32-vsqrt",
         "f32-vcmul", "f32-vrndne", "f32-f16-vcvt", "f32-vlrelu", "qs8-rsum", "qu8-rsum", "u8-vclamp", "s8-vclamp",
         "qu8-vadd-minmax", "qs8-vadd-minmax", "qs8-vcvt", "qs8-vlrelu", "qs8-vmul-minmax-fp32", "qu8-f32-vcvt",
         "qs8-f32-vcvt", "f32-argmaxpool", "f32-dwconv-minmax", "f32-dwconv2d-chw", "f32-gemm-minmax",
         "f32-igemm-minmax", "f32-maxpool", "qd8-f32-qc4w-gemm-minmax", "qs8-rdsum", "qu8-rdsum",
         "x32-packw-gemm-goi", "x32-transposec", "f32-conv-hwc2chw"]
NAMES = {"output_pixels": "pixel count", "pooling_elements": "pooling elements", "kernel_elements": "kernel elements",
         "input_height": "input height", "input_width": "input width", "output_channels": "output channels",
         "output_width": "output width", "block_width": "block width", "block_height": "block height"}
EVERY = {"output_pixels": "every pixel count", "output_width": "every output width", "channels": "every channel count"}
STOPS = ("COUNTEREXAMPLE", "MEMORY", "FIXTURE_ERROR", "CANDIDATE_REJECTS", "HARNESS_ERROR", "ERROR", "NO_RESULT")


def region(s):
    """growth_sweep's complete_region, e.g. "mr=1..4; nc=1..20; kc=1..20 (x4 bytes)" -> "mr 1..4; nc, kc 1..20"."""
    groups = []                                              # [text, range or None for "every"]
    for part in re.sub(r"\s*\([^)]*\)", "", s).split(";"):
        part = part.strip()
        if not part or " in [" in part:                      # harness variant
            continue
        m = re.fullmatch(r"(\w+)\s*(?:=|\s)\s*(?:(\d+)\.\.(\d+)|(\d+)|every value)", part)
        if not m:
            groups.append([part, None])
            continue
        name, lo, hi, single = m.groups()
        rng = None if lo is None and single is None else (single or (lo if lo == hi else f"{lo}..{hi}"))
        label = NAMES.get(name, name.replace("_", " "))
        if rng is None:
            groups.append([EVERY.get(name, f"every {label}"), None])
        elif groups and groups[-1][1] == rng:
            groups[-1][0] += ", " + label
        else:
            groups.append([label, rng])
    return "; ".join(t if r is None else f"{t} {r}" for t, r in groups)


def load(inputs):
    """{kernel: {tag: bound text}} from summary files (or directories searched for them); later inputs win."""
    out = {}
    for p in map(Path, inputs):
        files = [p] if p.is_file() else sorted(set(p.rglob("summary.json")) | set(p.rglob("*.summary.json")),
                                               key=lambda f: (len(f.parts), str(f)), reverse=True) if p.is_dir() else []
        if not files:
            print(f"bounds_table: skipping {p} (no summary.json)", file=sys.stderr)
        for f in files:                                      # a directory's own summary.json last (it wins)
            try:
                data = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):                       # elementwise: {kernel[@tag]: record}
                for spec, r in data.items():
                    if isinstance(r, dict) and "max_verified" in r:
                        k, _, tag = spec.partition("@")
                        out.setdefault(k, {})["" if tag == "ref" else tag] = elementwise_bound(r)
            elif isinstance(data, list):                     # growth_sweep: [record per kernel]
                for r in data:
                    if isinstance(r, dict) and r.get("kernel") and "completed_level" in r:
                        out[r["kernel"]] = {"": multidim_bound(r)}
    return out


def elementwise_bound(r):
    n = r.get("max_verified") or 0
    stop = next((s for s in r.get("stopped") or [] if s.get("status") in STOPS), None)
    if stop:
        return f"{stop['status']} at size {stop.get('batch', n + 1)}" + (f" (1..{n} verified)" if n else "")
    if r.get("error") and not n:
        return f"ERROR: {str(r['error'])[:60]}"
    return str(n) if n else "-"


def multidim_bound(r):
    st = r.get("status", "")
    if st in STOPS or not r.get("complete_region"):
        shape = r.get("stopped_shape") or (r.get("failure") or {}).get("shape")
        return f"{st or 'NO_RESULT'} at level {r.get('stopped_level', r.get('completed_level', 0) + 1)}" + \
               (f" ({', '.join(f'{a}={b}' for a, b in shape.items())})" if shape else "")
    return region(r["complete_region"])


def cell(tags):
    if tags is None:
        return "-"
    if "multeq" in tags or "multne" in tags:
        eq, ne = tags.get("multeq", "-"), tags.get("multne", "-")
        return f"{eq} (a_multiplier == b_multiplier) / {ne} (!=)"
    if "neon" in tags:
        return f"{tags[''] } (against Neon: {tags['neon']})" if "" in tags else f"{tags['neon']} (against Neon)"
    return tags.get("") or next(iter(tags.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlen", type=int, default=512)
    ap.add_argument("--phase1", nargs="+", type=Path, help="Phase-1 summaries (files or directories)")
    ap.add_argument("--phase2", nargs="+", type=Path, help="Phase-2 summaries (files or directories)")
    ap.add_argument("--csv", type=Path, help="also write the table to this CSV file")
    a = ap.parse_args()
    ew = "" if a.vlen == 512 else f"_vlen{a.vlen}"
    md = paths.OUT / ("multidim" if a.vlen == 512 else f"multidim-vlen{a.vlen}")
    p1 = load(a.phase1 or [paths.OUT / f"elementwise_phase1{ew}", md / "phase1"])
    p2 = load(a.phase2 or [paths.OUT / f"elementwise_phase2{ew}", md / "phase2"])
    kernels = [k for k in ORDER if k in p1 or k in p2] + sorted((set(p1) | set(p2)) - set(ORDER))
    rows = [(k, cell(p1.get(k)), cell(p2.get(k))) for k in kernels]
    print("| kernel | Phase-1 bound | Phase-2 bound |\n|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print(f"\n{len(rows)} kernels (Phase 1: {len(p1)}, Phase 2: {len(p2)})", file=sys.stderr)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "phase1_bound", "phase2_bound"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
