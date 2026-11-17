"""Elementwise bounded verification, fully symbolic: every input bit pattern, symbolic parameters within
XNNPACK's constraints, VLEN 512 (--vlen: any power of two >= 128), sizes 1, 2, 3, ... until the budget runs
out. The bound is the largest n such that every size 1..n verified. A second, execute-only pass measures line and branch coverage of the
reference and candidate over exactly the verified sizes.

  Phase 1: Neon reference vs Phase-1 RVV translation (default kernels/phase1/vlen<V>).
  Phase 2: Phase-1 RVV baseline ($SALT_PHASE1_DIR, default the reference kernels/phase2/vlen<V>/manifest.json
           names) vs optimized kernel (default kernels/phase2/vlen<V>), under the Phase-1 comparison rule, so the
           two proofs compose (Neon == baseline == optimized for every size within both bounds).

A kernel can carry a tag, kernel@tag:
  @ref     XNNPACK's NaN rule (any output accepted where the reference is NaN); Phase 1 uses it for vmax/vmin.
  @neon    Phase 2 against the Neon reference instead of the baseline (the pure reductions, rsum).
  @multeq / @multne   qu8-vadd's two cases of a_multiplier == b_multiplier, each proved separately.

usage: elementwise.py --phase {1,2} [--vlen 512] [--seconds 600] [--workers N] [--rvv-dir DIR] [kernel[@tag] ...]
Results: $SALT_AE_OUT/elementwise_phase<N>/summary.json (elementwise_phase<N>_vlen<V>/ for a VLEN other than
512) and one directory per kernel."""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths
from src.workflow.verification.engine import coverage
from src.workflow.verification.engine.emit import generate_harness
from src.workflow.verification.orchestrator import compile_harness

VLEN = 512                       # the paper's; --vlen changes it
DEFAULT = {"1": [k + "@ref" if k in ("f32-vmax", "f32-vmin") else k for k in paths.ELEMENTWISE],
           "2": [k + "@neon" if k.endswith("-rsum") else k for k in paths.ELEMENTWISE]}
FAILED = ("COUNTEREXAMPLE", "TIMEOUT", "MEMORY", "FIXTURE_ERROR", "CANDIDATE_REJECTS", "NATIVE_ACCESS")


def rvv_vlmax(src, vlen=VLEN):
    best = 0
    for sew, frac, lmul in re.findall(r"__riscv_vsetvl(?:max)?_e(\d+)m(f?)(\d+)\s*\(", src):
        best = max(best, int(vlen // int(sew) * (1 / int(lmul) if frac else int(lmul))))
    return best


def out_dir(phase, vlen=VLEN):
    """Where a phase's results go: elementwise_phase<N> at VLEN 512, elementwise_phase<N>_vlen<V> otherwise."""
    return paths.OUT / (f"elementwise_phase{phase}" + ("" if vlen == VLEN else f"_vlen{vlen}"))


def harness(phase, k, tag, rvv, vlen):
    if phase == "1" or tag == "neon":
        return generate_harness(str(paths.SOURCE / f"{k}.c"), str(rvv), k, params_mode="symbolic", input_range=None)
    st = "xnn_qu8_add_minmax_params"
    eq = (f"ctx.equal(_b_params.loadScalar(offsetof({st}, scalar.a_multiplier), 32), "
          f"_b_params.loadScalar(offsetof({st}, scalar.b_multiplier), 32))")
    case = {"multeq": [eq], "multne": [f"ctx.lnot_({eq})"]}.get(tag, [])
    return generate_harness(str(paths.phase1_reference(vlen) / f"{k}.c"), str(rvv), k, params_mode="symbolic", input_range=None,
                            reference_isa="rvv", rvv_bit_exact=False, param_case=case)


def one(job):
    phase, spec, rvv_dir, seconds, out, vlen = job
    k, _, tag = spec.partition("@")
    d = out / spec
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rvv = rvv_dir / f"{k}.c"
    res = {"rvv_file": str(rvv),
           "reference": "Neon" if phase == "1" or tag == "neon" else "Phase-1 RVV",
           "nan_rule": "XNNPACK (ref NaN = any)" if tag == "ref" else "strict (NaN==NaN, +0==-0)",
           "vlen": vlen, "rvv_vlmax": rvv_vlmax(rvv.read_text(), vlen)}
    try:
        src = harness(phase, k, tag, rvv, vlen)
    except Exception as e:
        return spec, dict(res, stage="generate", error=f"{type(e).__name__}: {e}"[:400])
    res["param_asserts"] = len(re.findall(r"ctx\.assert_\(.*_b_params", src))
    b, err = compile_harness(src, k, output_dir=d, backend="cvc5", extra_cmake_args=coverage.CMAKE_ARGS)
    if not b:
        return spec, dict(res, stage="compile", error=[l for l in err.splitlines() if "error:" in l][:4] or err[-600:])
    env = dict(os.environ, **paths.SOLVER_ENV, LLVM_PROFILE_FILE=str(d / "verify.profraw"))
    env.pop("SALT_NAN_POLICY", None)
    if tag == "ref":
        env["SALT_NAN_POLICY"] = "ref"
    t0 = time.time()
    try:
        p = subprocess.run([str(b), "1", "0", str(vlen), str(seconds)], capture_output=True, text=True,
                           timeout=seconds + 600, env=env)
        stdout, stderr, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        rc = None
        stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
        stderr = "hard timeout"
    (d / "events.jsonl").write_text(stdout)
    (d / "verify.profraw").unlink(missing_ok=True)
    ev = [json.loads(l) for l in stdout.splitlines() if l.startswith("{")]
    bad = {e.get("batch") for e in ev if e.get("status") in FAILED}
    ok = {e["batch"] for e in ev if e.get("status") == "VERIFIED"} - bad     # every path of a size verified
    n = 0
    while n + 1 in ok:
        n += 1
    stop = [e for e in ev if e.get("status") not in ("VERIFIED", "ALL_VERIFIED", "SOLVING", "NONVACUOUS", "PATHS")]
    if rc is not None and rc < 0:            # killed by a signal (a crash): sizes past the last event did not run
        stop.insert(0, {"status": "CRASH", "signal": -rc, "batch": n + 1})
    res.update(secs=round(time.time() - t0), max_verified=n,
               methods=sorted({e.get("method", "") for e in ev if e.get("status") == "VERIFIED"}),
               max_paths=max([e.get("paths", 1) for e in ev if e.get("status") == "PATHS"] or [1]),
               stopped=stop[:1], stderr_tail=stderr.strip().splitlines()[-1][:200] if stderr.strip() else "")
    if n:
        prof = d / "cov.profraw"
        envc = dict(env, SALT_EXECUTE_ONLY="1", LLVM_PROFILE_FILE=str(prof))
        try:
            subprocess.run([str(b), "1", str(n), str(vlen), "0"], capture_output=True, text=True, timeout=3600, env=envc)
            cov = coverage.summarize(b, [prof] if prof.exists() else [], d / "harnesses" / f"verify_{k}.cpp", d)
            res["coverage"] = {s: {x: cov[s][x] for x in ("lines_executed", "lines_instrumented", "branch_outcomes_covered",
                                                          "branch_outcomes", "partially_covered_branches", "unexecuted_lines")}
                               for s in ("reference", "candidate") if s in cov}
        except Exception as e:
            res["coverage"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    shutil.rmtree(d / "cmake_build", ignore_errors=True)
    return spec, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["1", "2"], required=True)
    ap.add_argument("--vlen", type=int, default=VLEN, help=f"VLEN in bits (default {VLEN})")
    ap.add_argument("--seconds", type=int, default=600, help="budget per kernel (default 600)")
    ap.add_argument("--workers", type=int, default=0,
                    help="kernels at once (default: one per physical core, or $SALT_JOBS)")
    ap.add_argument("--rvv-dir", type=Path, default=None,
                    help="candidate kernels (default: kernels/phase<N>/vlen<V>)")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    if a.vlen < 128 or a.vlen & (a.vlen - 1):
        ap.error("--vlen must be a power of two >= 128")
    specs = a.kernels or DEFAULT[a.phase]
    rvv_dir = (a.rvv_dir or paths.kernel_set(a.phase, a.vlen)).resolve()
    out = out_dir(a.phase, a.vlen)
    out.mkdir(parents=True, exist_ok=True)
    summ = out / "summary.json"
    res = json.loads(summ.read_text()) if summ.exists() else {}
    workers = min(a.workers or paths.physical_cores(), len(specs))
    print(f"Phase {a.phase}: {len(specs)} kernels at VLEN {a.vlen}, {a.seconds} s each, {workers} at once, "
          f"candidates from {rvv_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for spec, r in pool.map(one, [(a.phase, s, rvv_dir, a.seconds, out, a.vlen) for s in specs]):
            res[spec] = r
            summ.write_text(json.dumps(res, indent=1))
            c = r.get("coverage", {})
            cs = " ".join(f"{s[:3]} {v['branch_outcomes_covered']}/{v['branch_outcomes']}B" for s, v in c.items()
                          if isinstance(v, dict) and "branch_outcomes" in v)
            print(f"{spec:26s} {r.get('stage', 'run'):8s} verified 1..{r.get('max_verified', '-')} in {r.get('secs', '-')}s "
                  f"(VLMAX {r['rvv_vlmax']}) {r.get('methods', '')} {cs} "
                  f"{json.dumps(r.get('stopped') or r.get('error', ''))[:160]}", flush=True)
    print(f"DONE {summ}", flush=True)


if __name__ == "__main__":
    main()
