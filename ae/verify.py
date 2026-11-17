"""Verify the provided kernels: Phase 1 against Neon, Phase 2 against Phase 1, at VLEN 128 and 512.

  ./ae.sh verify                                       every kernel, both phases, both VLENs
  ./ae.sh verify --vlen 512 --phase 2 --kernels f32-vadd,f32-gemm-minmax
  ./ae.sh verify --seconds 60 --jobs 4                 a shorter budget, four cores
  ./ae.sh verify --redo                                run again the runs that already finished

Each kernel set is kernels/phase<P>/vlen<V>/ with a manifest.json: per kernel its family (elementwise or
multidim), its proofs (tagged specs, e.g. qu8-vadd-minmax@multeq) and the expected bound. Phase 1 is checked
against the Neon source (kernels/source); Phase 2 against the manifest's reference directory of Phase-1 kernels.
Elementwise kernels go through artifact/elementwise.py, multidimensional ones through growth_sweep. One run per
kernel, on one physical core, fully symbolic. Every proof gets the budget (default 600 s, the paper's); a kernel
with two proofs (the vadd kernels' two parameter cases at VLEN 512, Phase 2) runs them one after the other.

Output: $SALT_AE_OUT (default out/): verify/vlen<V>/phase<P>/<kernel>/ per run, and per VLEN bounds_vlen<V>.csv
(artifact/bounds_table.py), printed next to the expected bounds. Rerunning the command continues
where it stopped: a run whose summary is complete is kept (--redo runs it again), an unfinished one starts over."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
ARTIFACT = REPO / "artifact"


def physical_cpus():
    """One logical CPU per physical core this process may use (hyperthread siblings dropped)."""
    seen, cpus = set(), []
    try:
        for cpu in sorted(os.sched_getaffinity(0)):
            t = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = ((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip())
            if key not in seen:
                seen.add(key)
                cpus.append(cpu)
    except (AttributeError, OSError):
        cpus = list(range(os.cpu_count() or 1))
    return cpus


def jobs_for(vlen, phase, kernels, seconds, out):
    d = REPO / f"kernels/phase{phase}/vlen{vlen}"
    man = json.loads((d / "manifest.json").read_text())
    jobs = []
    for k, m in man["kernels"].items():
        if kernels and k not in kernels:
            continue
        o = out / f"vlen{vlen}/phase{phase}" / k
        env = {"SALT_AE_OUT": str(o)}
        if m["family"] == "elementwise":         # the kernel's proofs one after the other (--workers 1)
            cmd = [PY, str(ARTIFACT / "elementwise.py"), "--phase", str(phase), "--vlen", str(vlen),
                   "--seconds", str(seconds), "--workers", "1", "--rvv-dir", str(d), *m["specs"]]
            if phase == 2:
                env["SALT_PHASE1_DIR"] = str(REPO / man["reference"])
            cwd = ARTIFACT
        else:
            cmd = [PY, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", str(vlen),
                   "--seconds", str(seconds), "--coverage", "--rvv-dir", str(d), "--kernel", k,
                   "--output", str(o / "sweep")]          # growth_sweep creates its output directory
            if phase == 2:
                cmd += ["--reference-dir", str(REPO / man["reference"])]
            cwd = REPO
        jobs.append(dict(vlen=vlen, phase=phase, kernel=k, family=m["family"], specs=m["specs"], cmd=cmd, env=env,
                         cwd=cwd, out=o,
                         seconds=seconds * len(m["specs"]), expected=m["expected_bound"]))
    return jobs


def finished(j):
    """The run's summary has every proof (elementwise) or the sweep wrote its summary (multidimensional)."""
    try:
        if j["family"] == "elementwise":
            sub = f"elementwise_phase{j['phase']}" + ("" if j["vlen"] == 512 else f"_vlen{j['vlen']}")
            got = json.loads((j["out"] / sub / "summary.json").read_text())
            return all(s in got and "max_verified" in got[s] for s in j["specs"])
        return (j["out"] / "sweep/summary.json").exists()
    except (OSError, ValueError):
        return False


def run_all(jobs, ncores):
    cpus = physical_cpus()[:ncores]
    free, lock = list(cpus), threading.Lock()
    taskset = shutil.which("taskset")
    done = [0]

    def run(j):
        with lock:
            cpu = free.pop(0)
        try:
            shutil.rmtree(j["out"], ignore_errors=True)        # an unfinished run starts over
            j["out"].mkdir(parents=True)
            cmd = ([taskset, "-c", str(cpu)] if taskset else []) + j["cmd"]
            t0 = time.time()
            with open(j["out"] / "run.log", "w") as log:
                rc = subprocess.run(cmd, cwd=j["cwd"], env=dict(os.environ, **j["env"]), stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(jobs)}] vlen {j['vlen']} phase {j['phase']} {j['kernel']}: exit {rc}, "
                      f"{time.time() - t0:.0f}s (core {cpu})", flush=True)
        finally:
            with lock:
                free.append(cpu)

    with ThreadPoolExecutor(max_workers=len(cpus)) as ex:          # the longest runs first
        list(ex.map(run, sorted(jobs, key=lambda j: -j["seconds"])))


def table(vlen, out, expected):
    p1, p2 = out / f"vlen{vlen}/phase1", out / f"vlen{vlen}/phase2"
    csv_path = out.parent / f"bounds_vlen{vlen}.csv"
    args = [PY, str(ARTIFACT / "bounds_table.py"), "--vlen", str(vlen), "--csv", str(csv_path)]
    if p1.exists():
        args += ["--phase1", str(p1)]
    if p2.exists():
        args += ["--phase2", str(p2)]
    subprocess.run(args, cwd=ARTIFACT, check=True, stdout=subprocess.DEVNULL)
    got = {r["kernel"]: r for r in csv.DictReader(open(csv_path))}
    print(f"\nVLEN {vlen}: verified bounds (this run, {csv_path} | expected, artifact/expected/final_vlen{vlen}_bounds.csv)")
    for k, r in got.items():
        e = expected.get(k, {})
        for ph in ("phase1", "phase2"):
            if r.get(f"{ph}_bound"):
                print(f"  {k:26s} {ph}: {r[f'{ph}_bound'][:60]:60s} | {e.get(f'{ph}_bound', '')[:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlen", default="128,512", help="comma-separated VLENs (default 128,512)")
    ap.add_argument("--phase", default="1,2", help="comma-separated phases (default 1,2)")
    ap.add_argument("--kernels", default="all", help="comma-separated kernel names, or all")
    ap.add_argument("--seconds", type=int, default=600, help="SMT budget per run (default 600, the paper's)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel runs, one per physical core (default: all cores)")
    ap.add_argument("--list", action="store_true", help="print the runs and exit")
    ap.add_argument("--redo", action="store_true", help="run again the runs that already finished")
    a = ap.parse_args()
    out = Path(os.environ.get("SALT_AE_OUT", REPO / "out")).resolve() / "verify"
    kernels = None if a.kernels == "all" else set(a.kernels.split(","))
    vlens, phases = [int(v) for v in a.vlen.split(",")], [int(p) for p in a.phase.split(",")]
    jobs = [j for v in vlens for p in phases for j in jobs_for(v, p, kernels, a.seconds, out)]
    if not jobs:
        sys.exit("no runs selected")
    ncores = a.jobs or len(physical_cpus())
    if a.list:
        for j in jobs:
            print(f"vlen {j['vlen']} phase {j['phase']} {j['kernel']}: {' + '.join(j['specs'])} ({j['seconds']} s)"
                  + ("  [finished]" if finished(j) else ""))
        return
    if not a.redo:
        done = [j for j in jobs if finished(j)]
        jobs = [j for j in jobs if not finished(j)]
        if done:
            print(f"{len(done)} runs already finished, kept (--redo to run them again)", flush=True)
    total = sum(j["seconds"] for j in jobs)
    if jobs:
        print(f"{len(jobs)} runs, {a.seconds}s per proof, {ncores} at once (about "
              f"{max(total / ncores, max(j['seconds'] for j in jobs)) / 60 * 1.1:.0f} min) -> {out}", flush=True)
        run_all(jobs, ncores)
    for v in vlens:
        exp = {r["kernel"]: r for r in csv.DictReader(open(ARTIFACT / f"expected/final_vlen{v}_bounds.csv"))}
        table(v, out, exp)


if __name__ == "__main__":
    main()
