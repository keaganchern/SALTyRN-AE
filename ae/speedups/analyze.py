"""Speedups over XNNPACK's RVV kernel (Upstream) from the FireSim uartlogs of `./ae.sh speedups run`.

  ./ae.sh speedups analyze [--vlen 128,512] [--logs DIR]

Speedup = Upstream cycles / lane cycles per element at the same batch; per kernel the geomean over the batches; the
GEOMEAN row is over the kernels. Every lane divides by the Upstream rows of its own ELF where that ELF has them (main:
Phase 1 and Phase 2; phase1 at VLEN 128: Phase 1), else by those of the main ELF. At VLEN 128 the Phase-1 column comes
from the phase1 ELF. f32-argmaxpool compares at pooling element count 25 (its pe=9
section has no Upstream lane); the Upstream-only -u2v/-u4v/-u8v sections are skipped. SIMDe: RVV mode, or portable
mode for the kernels RVV mode crashes on.

Input: $SALT_AE_OUT/speedups/vlen<V>/firesim/<group>.uartlog (or --logs DIR/<group>.uartlog).
Output: $SALT_AE_OUT (default out/) /speedups_vlen<V>.csv (the columns of artifact/expected/final_vlen<V>_speedups.csv),
and next to the uartlogs' directory speedups_vlen<V>_by_batch.csv and speedups_vlen<V>.md; the table is printed next to
the expected one."""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAME = {"qu8-vadd": "qu8-vadd-minmax", "qs8-vadd": "qs8-vadd-minmax", "qs8-vmul": "qs8-vmul-minmax-fp32"}
COLS = ["phase1", "phase2", "gcc_autovec", "neon2rvv", "simde"]
LANE = {("", "Sol"): "phase1", ("-pe25", "Sol"): "phase1", ("", "Autocomp"): "phase2", ("-pe25", "Autocomp"): "phase2",
        ("-autovec", "Autocomp"): "gcc_autovec", ("-neon2rvv", "Neon2RVV"): "neon2rvv", ("-simde", "SIMDe"): "simde"}
geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def rows(path):
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("CSV:"):
            f = line[4:].strip().split(",")
            if len(f) == 7:
                yield f[0], f[1], int(f[2]), float(f[5])


def section(label):
    m = re.fullmatch(r"(.+?)(-autovec|-neon2rvv|-pe25|-simde)?", label)
    return NAME.get(m[1], m[1]), m[2] or ""


def analyze(vlen, logs, root):
    up = defaultdict(lambda: defaultdict(dict))      # group -> kernel -> batch -> Upstream cycles/elem
    cpe = defaultdict(dict)                          # (kernel, column) -> batch -> (cycles/elem, group)
    simde_mode = {}
    found = sorted(p for p in logs.glob("*.uartlog"))
    if not any(p.stem == "main" for p in found):
        sys.exit(f"no main.uartlog in {logs}")
    own_p1 = any(p.stem == "phase1" for p in found)          # VLEN 128: Phase 1 timed in its own ELF
    for p in sorted(found, key=lambda p: (p.stem != "simde-rvv", p.stem)):    # SIMDe RVV mode before portable
        g = p.stem
        for label, impl, batch, c in rows(p):
            k, suffix = section(label)
            if re.search(r"-u[248]v$", k) or (k == "f32-argmaxpool" and suffix == ""):
                continue
            if impl == "Upstream" and suffix in ("", "-pe25"):
                up[g][k][batch] = c
                continue
            col = LANE.get((suffix, impl))
            if not col or (col == "phase1" and own_p1 and g != "phase1"):
                continue
            if col == "simde":
                mode = "RVV" if g == "simde-rvv" else "portable"
                if simde_mode.setdefault(k, mode) != mode:
                    continue
            cpe[(k, col)][batch] = (c, g)
    kernels = sorted({k for k, _ in cpe})
    table, detail = [], []
    for k in kernels:
        row = {"kernel": k}
        for col in COLS:
            sp = []
            for b, (c, g) in sorted(cpe.get((k, col), {}).items()):
                u = (up[g][k] if up[g].get(k) else up["main"][k]).get(b)
                if u and c > 0:
                    sp.append(u / c)
                    detail.append({"kernel": k, "lane": col, "batch": b, "cycles_per_elem": c,
                                   "upstream_cycles_per_elem": u, "speedup": round(u / c, 4), "elf": g})
            row[col] = geo(sp)
        row["simde_mode"] = simde_mode.get(k, "") if row["simde"] else ""
        table.append(row)
    table.append({"kernel": "GEOMEAN", **{c: geo([r[c] for r in table if r[c]]) for c in COLS}, "simde_mode": ""})
    out = logs.parent
    f3 = lambda v: f"{v:.3f}" if isinstance(v, float) else v
    main_csv = root / f"speedups_vlen{vlen}.csv"
    root.mkdir(parents=True, exist_ok=True)
    with open(main_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", *COLS, "simde_mode"])
        w.writeheader()
        w.writerows({k: f3(v) for k, v in r.items()} for r in table)
    if detail:
        with open(out / f"speedups_vlen{vlen}_by_batch.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0]))
            w.writeheader()
            w.writerows(detail)
    fx = lambda v: f"{v:.2f}x" if v else "-"
    md = ["| kernel | Phase 1 | Phase 2 | GCC autovec | Neon2RVV | SIMDe |", "|---|---|---|---|---|---|"]
    md += [f"| {r['kernel']} | " + " | ".join(fx(r[c]) for c in COLS) + " |" for r in table]
    (out / f"speedups_vlen{vlen}.md").write_text("\n".join(md) + "\n")

    exp_path = REPO / f"artifact/expected/final_vlen{vlen}_speedups.csv"
    exp = {r["kernel"]: r for r in csv.DictReader(open(exp_path))} if exp_path.exists() else {}
    print(f"\nVLEN {vlen}: speedup over Upstream, this run (expected: {exp_path.relative_to(REPO)})")
    print(f"  {'kernel':26s}" + "".join(f"{c:>18s}" for c in COLS))
    for r in table:
        e = exp.get(r["kernel"], {})
        cell = lambda c: (f"{r[c]:.2f}" if r[c] else "-") + " (" + (f"{float(e[c]):.2f}" if e.get(c) else "-") + ")"
        print(f"  {r['kernel']:26s}" + "".join(f"{cell(c):>18s}" for c in COLS))
    print(f"  -> {main_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlen", default="128,512", help="comma-separated VLENs (default 128,512)")
    ap.add_argument("--logs", help="directory of <group>.uartlog files (one VLEN; default: the run's)")
    a = ap.parse_args()
    root = Path(os.environ.get("SALT_AE_OUT", REPO / "out")).resolve()
    for v in (int(x) for x in a.vlen.split(",")):
        logs = Path(a.logs).resolve() if a.logs else root / f"speedups/vlen{v}/firesim"
        analyze(v, logs, logs.parent if a.logs else root)


if __name__ == "__main__":
    main()
