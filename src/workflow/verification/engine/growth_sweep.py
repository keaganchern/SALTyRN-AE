"""Grow every multidimensional kernel's shapes level by level, one solver process per shape.

Run from the repository root:
  .venv/bin/python -m src.workflow.verification.engine.growth_sweep --vlen 512 --seconds 600
  ... --rvv-dir <dir of RVV kernels>        # verify another set of kernels

Each kernel's shape domain comes from its source (shape_plan.py): sizes, their
units and bounds from the reference's entry asserts, everything else from the
fixture's ABI. Level N verifies every shape whose largest axis index is N, with
every variant (offsets, padding, parameter profiles), so completed levels 1..N
cover the whole box [1..N]^k with no gaps. Each kernel stops at its first shape
that does not verify. Kernels run concurrently, compilation finishes before a
shared start barrier, and no result is cached.

A shape counts as VERIFIED only on an explicit VERIFIED event and exit status 0.
Under the default checks a pass also means: the candidate touched no memory
outside the reference's footprint (SALT_MEMCHECK=strict), and, at level 1, the
fixture's assumptions are satisfiable and the reference output is not constant
(SALT_NONVACUITY=1). UF-first solving (SALT_UF_FIRST) proves shapes whose FP
operation graphs coincide without bit-blasting them; it is sound both ways and
falls back to the exact query.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from . import coverage
from .cutoff import lane_split, outer_loop
from .emit import generate_shape_harness
from .shape_plan import probe_shapes
from ..orchestrator import PROJECT_ROOT, BITWUZLA_BIN, compile_harness

# The paper's multidimensional kernels (names as in kernels/source).
MULTIDIM = ("qs8-rdsum", "qu8-rdsum", "f32-argmaxpool", "f32-maxpool", "f32-dwconv-minmax",
            "f32-dwconv2d-chw", "f32-conv-hwc2chw", "f32-gemm-minmax", "f32-igemm-minmax",
            "qd8-f32-qc4w-gemm-minmax", "x32-packw-gemm-goi", "x32-transposec")
# File names used by the Phase-2 kernel directory for the same kernels.
ALIASES = {"f32-dwconv2d-chw": "dwconv2d-chw", "x32-transposec": "x32transpose"}
INFO = {"SOLVING", "NONVACUOUS", "NONVACUITY_UNKNOWN", "MEMORY_WARNING", "PATHS"}
FAILURES = ("COUNTEREXAMPLE", "MEMORY", "CANDIDATE_REJECTS", "VACUOUS_ASSUMPTIONS",
            "VACUOUS_CONSTANT_OUTPUT", "NATIVE_ACCESS")


def candidate_path(rvv_dir: Path, name: str) -> Path:
    for n in (name, ALIASES.get(name, name)):
        if (rvv_dir / f"{n}.c").is_file():
            return rvv_dir / f"{n}.c"
    raise FileNotFoundError(f"no candidate for {name} in {rvv_dir}")


def classify(returncode, events):
    statuses = [e.get("status") for e in events]
    for f in FAILURES:
        if f in statuses:
            return f
    rest = [s for s in statuses if s not in INFO]
    if returncode == 0 and rest == ["VERIFIED"] and statuses[-2:] == ["SOLVING", "VERIFIED"]:
        return "VERIFIED"
    # A shape whose C-level branches forked: one VERIFIED per path, each after its SOLVING, then the PATHS count.
    paths = [e.get("paths") for e in events if e.get("status") == "PATHS"]
    if returncode == 0 and len(paths) == 1 and isinstance(paths[0], int) and paths[0] > 1 and \
            statuses[-1] == "PATHS" and rest == ["VERIFIED"] * paths[0] and \
            all(statuses[i - 1] == "SOLVING" for i, s in enumerate(statuses) if s == "VERIFIED"):
        return "VERIFIED"
    if returncode == 2 and "TIMEOUT" in statuses:
        return "TIMEOUT"
    return "ERROR"


def execute_shape(command, seconds, env):
    start = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=max(0.001, seconds - (time.monotonic() - start)))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):   # the group already exited (macOS reports EPERM)
            pass
        stdout, stderr = proc.communicate()
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
        except json.JSONDecodeError:
            pass
    elapsed = time.monotonic() - start
    status = classify(proc.returncode, events)
    if status not in FAILURES and (timed_out or elapsed > seconds):
        status = "TIMEOUT"
    method = next((e.get("method") for e in events if e.get("status") == "VERIFIED"), None)
    return dict(status=status, seconds=elapsed, returncode=proc.returncode, method=method,
                events=events, stderr=stderr[-2000:], hard_deadline=timed_out)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def induction(plan, rvv_src, vlen, ref_src=None, spec=None):
    """Cap the axis the candidate's outermost loop runs over at its stride.

    If that loop meets the premise of cutoff.outer_loop and no derived stride
    depends on the dimension, sizes 1..stride (for every value of the other axes
    in a complete level) prove every size by induction on the outer loop. The
    axis keeps at least 2 values, an empirical check of the induction step.
    The base cases go to 2S, so a second iteration checks that both kernels
    shift their pointers alike. If the reference's outer loop meets the same premise with a stride dividing
    the candidate's, the induction needs nothing about the reference; otherwise
    it assumes the reference implements the operator, parallel in that dimension."""
    ol = outer_loop(rvv_src, plan.derived)
    axis = next((a for a in plan.axes if a.name == ol.dim), None) if ol.ok else None
    if not axis or axis.step != 1:
        return {"applies": False, "reason": ol.reason if not ol.ok else f"{ol.dim} is not a unit axis"}
    if any(re.search(rf"\b{re.escape(ol.dim)}\b", e) for e in plan.derived.values()):
        return {"applies": False, "reason": f"a derived stride depends on {ol.dim}"}
    stride = ol.units(vlen)
    # The remaining instance must start at a weight-tile boundary on the NEON side.
    for d in (spec.dual_weights.values() if spec else ()):
        if d["ncols"] == ol.dim and str(d["neon_tile"]).isdigit() and stride % int(d["neon_tile"]):
            return {"applies": False, "reason": f"stride {stride} is not a multiple of the NEON weight tile"}
    # The step assumes both kernels shift their pointers to the same logical
    # instance; the syntax check cannot see that. Sizes up to 2S run a second
    # iteration on both its full and its tail path, which checks the shift.
    cap = 2 * stride
    if axis.hi is None or axis.hi > cap:
        axis.hi = cap
    assumption = "reference implements the operator for every size, parallel in " + ol.dim
    if ref_src is not None:
        ref = outer_loop(re.sub(r"\btest_neon\b", "test_rvv", ref_src), plan.derived)
        # R(D) = S / S_R full reference iterations, then R(D - S): its stride must divide ours.
        if ref.ok and ref.dim == ol.dim and ref.units(vlen) and stride % ref.units(vlen) == 0:
            assumption = "none: the reference's outer loop meets the same premise"
    out = {"applies": True, "dim": ol.dim, "stride": stride, "cap": axis.hi, "assumption": assumption}
    # Within one outer iteration, a channel loop that splits lane by lane on both
    # sides (cutoff.lane_split) needs only 1..T channels; the outer induction then
    # extends every channel count to every outer size.
    for a in plan.axes if ref_src is not None else ():
        if a.name == ol.dim or a.step != 1:
            continue
        ls = lane_split(rvv_src, ref_src, a.name, plan.derived)
        # Packed weights over these channels: the reference's layout must split at T
        # (the candidate's lanes read their own channel's weights by the packing contract).
        tiles = [d["neon_tile"] for d in (spec.dual_weights.values() if spec else ()) if d["ncols"] == a.name]
        if ls.ok and any(not str(t).isdigit() or ls.base % int(t) for t in tiles):
            continue
        if ls.ok:                     # 1..T suffice; up to 2T also runs a second reference iteration
            if a.hi is None or a.hi > 2 * ls.base:
                a.hi = 2 * ls.base
            out["lanes"] = {"dim": a.name, "base": ls.base, "cap": a.hi}
            break
    return out


def prepare(name, rvv_dir, build_root, input_range, params_symbolic, with_coverage=False,
            vlen=512, use_induction=True, reference_dir=None):
    neon = PROJECT_ROOT / "kernels/source" / f"{name}.c"
    if reference_dir:
        # Phase 2 against Phase 1: a (verified) Phase-1 RVV kernel takes the reference's place. Its entry point is
        # renamed into the reference slot; the fixture, footprint and comparison are the Neon check's.
        ref_src = candidate_path(Path(reference_dir), name).read_text()
        neon = build_root / "reference" / f"{name}.c"
        neon.parent.mkdir(parents=True, exist_ok=True)
        neon.write_text(re.sub(r"\btest_rvv\b", "test_neon", ref_src))
    rvv = candidate_path(rvv_dir, name)
    # Tolerance is embedded during generation, so runtime sanitization is too late.
    old_tolerance = os.environ.pop("SALT_ULP_TOL", None)
    try:
        source, spec, plan = generate_shape_harness(str(neon), str(rvv), name, input_range, params_symbolic)
    finally:
        if old_tolerance is not None:
            os.environ["SALT_ULP_TOL"] = old_tolerance
    binary, error = compile_harness(source, name, output_dir=build_root / name, backend="cvc5",
                                    extra_cmake_args=coverage.CMAKE_ARGS if with_coverage else None)
    if not binary:
        raise RuntimeError(error)
    probes = probe_shapes(plan, [neon.read_text(), rvv.read_text()], vlen=vlen)   # before induction caps the axes
    ind = (induction(plan, rvv.read_text(), vlen, neon.read_text(), spec) if use_induction
           else {"applies": False, "reason": "disabled"})
    return plan, dict(binary=str(binary), rvv=str(rvv), dimensions=list(spec.dims),
                      harness=str(build_root / name / "harnesses" / f"verify_{name}.cpp"),
                      plan=plan.describe(), induction=ind, probes=probes,
                      weights={n: dict(d) for n, d in spec.dual_weights.items()},
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      neon_sha256=hashlib.sha256(neon.read_bytes()).hexdigest(),
                      rvv_sha256=hashlib.sha256(rvv.read_bytes()).hexdigest())


def complete_region(plan, level, ind=None):
    """Human-readable statement of the region levels 1..level cover."""
    parts = []
    for a in plan.axes:
        top = min(level, a.hi) if a.hi else level
        unit = f" (x{a.step} bytes)" if a.step > 1 else ""
        if ind and ind.get("applies") and a.name == ind["dim"] and level >= ind["cap"]:
            parts.append(f"{a.name}=every value (outer-loop induction, stride {ind['stride']}; base 1..{ind['cap']} verified)")
            continue
        lanes = (ind or {}).get("lanes")
        if lanes and a.name == lanes["dim"] and level >= max(lanes["cap"], ind["cap"]):
            parts.append(f"{a.name}=every value (lane decomposition, T={lanes['base']}; base 1..{lanes['cap']} verified)")
            continue
        parts.append(f"{a.name}=1..{top}{unit}")
    parts += [f"{n} in {v}" for n, v in plan.variants.items()]
    return "; ".join(parts)


def run_probes(plan, prepared, target, result, deadline, vlen, base_env, profiles, log):
    """Verify the threshold probes (shape_plan.probe_shapes) the completed box does
    not contain, until `deadline`. A probe that fails fails the kernel; one that
    times out is recorded as unverified."""
    level = result["completed_level"]
    inside = lambda s: all(s[a.name] // a.step <= (min(level, a.hi) if a.hi else level) for a in plan.axes)
    todo = [s for s in prepared.get("probes", []) if not inside(s)]
    out = dict(planned=len(prepared.get("probes", [])), in_box=len(prepared.get("probes", [])) - len(todo),
               verified=[], unverified=[])
    result["probes"] = out
    for i, shape in enumerate(todo):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            out["unverified"] += [dict(shape=s, status="NOT_RUN") for s in todo[i:]]
            break
        limit = max(1.0, remaining / 2)           # one probe may not take the whole share
        env = dict(base_env)
        profile = target / "prof" / f"p{i}.profraw"
        env["LLVM_PROFILE_FILE"] = str(profile)
        command = [prepared["binary"], str(vlen), str(limit)] + [str(shape[d]) for d in prepared["dimensions"]]
        record = execute_shape(command, limit, env)
        record.update(shape=shape, probe=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
        if record["status"] == "VERIFIED":
            out["verified"].append(shape)
            result["methods"][record["method"]] = result["methods"].get(record["method"], 0) + 1
            if profile.exists():
                profiles.append(profile)
            continue
        profile.unlink(missing_ok=True)
        if record["status"] in FAILURES:
            result.update(status=record["status"], stopped_shape=shape, failure=record)
            out["unverified"].append(dict(shape=shape, status=record["status"]))
            break
        out["unverified"].append(dict(shape=shape, status=record["status"]))


def run_kernel(name, plan, prepared, outdir, seconds, vlen, barrier, max_level, base_env, nonvacuity,
               with_coverage=False, probe_share=0.25):
    target = outdir / name
    target.mkdir()
    result = dict(kernel=name, vlen=vlen, budget_seconds=seconds, status="RUNNING",
                  verified_shapes=0, completed_level=0, attempted_shapes=0, methods={},
                  plan=prepared["plan"], induction=prepared.get("induction"), rvv=prepared["rvv"],
                  last_verified_shape=None)
    write_json(target / "result.json", result)
    barrier.wait()
    start = time.monotonic()
    result["started_utc"] = datetime.now(timezone.utc).isoformat()
    final = start + seconds
    # The box gets the budget but a share kept for the threshold probes.
    deadline = start + seconds * (1 - probe_share) if prepared.get("probes") and probe_share > 0 else final
    profiles = []                      # coverage profiles of VERIFIED shapes only
    (target / "prof").mkdir()
    with (target / "shapes.jsonl").open("w") as log:
        for level in itertools.count(1):
            if plan.exhausted(level):
                result["status"] = "COMPLETE"   # every axis bounded and fully verified
                break
            for shape in plan.frontier(level):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result.update(status="BUDGET_EXHAUSTED", stopped_shape=shape, stopped_level=level)
                    break
                result.update(current_shape=shape, current_level=level)
                write_json(target / "result.json", result)
                env = dict(base_env)
                if nonvacuity == "all" or (nonvacuity == "level1" and level == 1):
                    env["SALT_NONVACUITY"] = "1"
                profile = target / "prof" / f"s{result['attempted_shapes']}.profraw"
                env["LLVM_PROFILE_FILE"] = str(profile)
                command = [prepared["binary"], str(vlen), str(remaining)] + [str(shape[d]) for d in prepared["dimensions"]]
                record = execute_shape(command, remaining, env)
                record.update(shape=shape, level=level, elapsed_seconds=time.monotonic() - start)
                result["attempted_shapes"] += 1
                log.write(json.dumps(record) + "\n")
                log.flush()
                if record["status"] != "VERIFIED":
                    profile.unlink(missing_ok=True)    # only verified shapes count for coverage
                    result.update(status=record["status"], stopped_shape=shape, stopped_level=level,
                                  failure=record)
                    break
                result["verified_shapes"] += 1
                if profile.exists():
                    profiles.append(profile)
                    if len(profiles) >= 25:            # keep disk use bounded
                        coverage.merge_into(target / "verified.profdata", profiles)
                        profiles.clear()
                result["methods"][record["method"]] = result["methods"].get(record["method"], 0) + 1
                result["last_verified_shape"] = shape
                result["elapsed_seconds"] = time.monotonic() - start
                write_json(target / "result.json", result)
            else:
                result["completed_level"] = level
                result["complete_region"] = complete_region(plan, level, prepared.get("induction"))
                write_json(target / "result.json", result)
                print(f"{name}: completed level {level}, {result['verified_shapes']} shapes", flush=True)
                if max_level and level >= max_level:
                    result["status"] = "MAX_LEVEL_REACHED"
                    break
                continue
            break
        if result["status"] not in FAILURES and prepared.get("probes") and probe_share > 0:
            run_probes(plan, prepared, target, result, final, vlen, base_env, profiles, log)
    result["elapsed_seconds"] = time.monotonic() - start
    if with_coverage:
        try:
            result["coverage"] = coverage.summarize(Path(prepared["binary"]), profiles,
                                                    Path(prepared["harness"]), target,
                                                    count=result["verified_shapes"])
        except Exception as e:                              # report, never mask the verdicts
            result["coverage"] = {"error": f"{type(e).__name__}: {e}"}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result.pop("current_shape", None)
    result.pop("current_level", None)
    write_json(target / "result.json", result)
    print(f"{name}: {result['status']}, completed level {result['completed_level']}, "
          f"{result['verified_shapes']} shapes", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlen", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--kernel", action="append", choices=MULTIDIM)
    parser.add_argument("--rvv-dir", type=Path, default=PROJECT_ROOT / "kernels/target",
                        help="candidate kernels (default: the Phase-1 targets)")
    parser.add_argument("--input-range", default=None,
                        help="restrict FP inputs to finite LO,HI (default: full domain)")
    parser.add_argument("--params", choices=["symbolic", "fixture"], default="symbolic",
                        help="f32 clamp bounds: symbolic under min <= max, neither NaN (default), or the fixture's profile")
    parser.add_argument("--memcheck", choices=["strict", "report", "off"], default="strict")
    parser.add_argument("--nan-policy", choices=["xnnpack", "both"], default="xnnpack",
                        help="xnnpack: an output whose reference value is NaN is don't-care (XNNPACK's "
                             "oracle for these kernels); both: it must be NaN on both sides")
    parser.add_argument("--nonvacuity", choices=["level1", "all", "off"], default="level1")
    parser.add_argument("--uf-first", default="mul,fma,add,sub,bvmul",
                        help="FP ops abstracted in the first solving stage ('' to disable)")
    parser.add_argument("--probe-share", type=float, default=0.25,
                        help="budget share for threshold probes outside the box (0: none)")
    parser.add_argument("--no-induction", action="store_true",
                        help="grow every axis in the box, even one an outer-loop induction covers")
    parser.add_argument("--coverage", action="store_true",
                        help="report line/branch coverage of the kernels over the verified shapes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-level", type=int, default=0, help="Only for finite smoke checks; zero keeps growing")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="verify against these (Phase-1) RVV kernels instead of the Neon sources")
    args = parser.parse_args()
    if args.vlen < 128 or args.vlen & (args.vlen - 1) or args.seconds <= 0 or args.max_level < 0:
        parser.error("VLEN must be a power of two >=128, seconds positive, max-level nonnegative")
    input_range = tuple(float(x) for x in args.input_range.split(",")) if args.input_range else None
    names = [n for n in MULTIDIM if not args.kernel or n in args.kernel]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output or PROJECT_ROOT / "results" / f"growth_vlen{args.vlen}_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    # One build directory per output directory: two sweeps whose outputs share a leaf name (a kernel at two
    # VLENs) would otherwise compile into the same binary path and run each other's harness.
    build_root = PROJECT_ROOT / "build/verification_cvc5" / f"{out.name}-{hashlib.sha1(str(out).encode()).hexdigest()[:8]}"
    env = os.environ.copy()
    for key in ("SALT_UF_ABSTRACT", "SALT_ULP_TOL", "SALT_CVC5_OPTIONS", "SALT_DUMP_SMT2", "SALT_STATS",
                "SALT_BACKEND", "SALT_BITWUZLA", "SALT_NONVACUITY", "SALT_UF_FIRST", "SALT_MEMCHECK",
                "SALT_NAN_POLICY"):
        env.pop(key, None)
    env["SALT_MEMCHECK"] = args.memcheck
    # XNNPACK never tests these kernels on NaN and accepts any output where its
    # binary-op reference is NaN; the unary oracle ("both") requires a NaN there.
    env["SALT_NAN_POLICY"] = "ref" if args.nan_policy == "xnnpack" else "both"
    if args.uf_first:
        env["SALT_UF_FIRST"] = args.uf_first
    # Use the same automatic portfolio as the ordinary verifier, where available.
    if BITWUZLA_BIN.is_file():
        env.update(SALT_BACKEND="auto", SALT_BITWUZLA=str(BITWUZLA_BIN))
    metadata = dict(vlen=args.vlen, budget_per_kernel_seconds=args.seconds, workers=len(names),
                    max_level=args.max_level, cpu_count=os.cpu_count(), compile_excluded=True,
                    rvv_dir=str(args.rvv_dir), input_range=input_range, params=args.params,
                    memcheck=args.memcheck, nan_policy=args.nan_policy,
                    nonvacuity=args.nonvacuity, uf_first=args.uf_first or None,
                    output_tolerance_override=None,
                    stop_policy="Stop each kernel at its first non-VERIFIED shape; completed levels have no gaps.",
                    solver="cvc5, automatic Bitwuzla routing for pure bit-vector queries",
                    files={})
    for folder in ("src/verification_cvc5", "src/workflow/verification/engine"):
        for p in sorted((PROJECT_ROOT / folder).rglob("*")):
            if p.suffix in (".hpp", ".py", ".h", ".txt"):
                metadata["files"][str(p.relative_to(PROJECT_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    plans, prepared, failed = {}, {}, {}
    for name in names:
        print(f"Preparing {name}...", flush=True)
        try:
            plans[name], prepared[name] = prepare(name, args.rvv_dir, build_root, input_range,
                                                  args.params == "symbolic", args.coverage,
                                                  args.vlen, not args.no_induction, args.reference_dir)
        except Exception as e:          # one kernel's harness error must not stop the others
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"{name}: HARNESS_ERROR {failed[name]}", flush=True)
    names = [n for n in names if n not in failed]
    metadata["prepared"] = prepared
    write_json(out / "manifest.json", metadata)
    print(f"All {len(names)} harnesses compiled. Starting {args.seconds:g}s per kernel at VLEN={args.vlen}. "
          f"Results: {out}", flush=True)
    results = []
    if names:
        barrier = threading.Barrier(len(names))
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            jobs = [pool.submit(run_kernel, n, plans[n], prepared[n], out, args.seconds, args.vlen, barrier,
                                args.max_level, env, args.nonvacuity, args.coverage, args.probe_share)
                    for n in names]
            results = [job.result() for job in jobs]
    results += [dict(kernel=n, vlen=args.vlen, status="HARNESS_ERROR", error=e, verified_shapes=0,
                     completed_level=0, methods={}) for n, e in failed.items()]
    write_json(out / "summary.json", results)
    print(f"Summary: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
