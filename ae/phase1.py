#!/usr/bin/env python3
"""SALTyRN Phase 1 on chosen kernels and VLENs, each translation then checked by the fully symbolic sweep.

  ./ae.sh phase1 --list                                  the 35 kernels, by family
  ./ae.sh phase1 --kernels f32-vadd,qs8-vcvt --vlen 512
  ./ae.sh phase1 --kernels all --vlen 128,512            everything, both vector lengths

Every kernel, at every VLEN, holds one core while it runs two steps:

  1  Phase 1 (salty-rn): translate the Neon kernel with Gemini, compile it and run it on Spike at that VLEN,
     then the pipeline's 600 s Neon-vs-RVV proof (edge-case parameter values). A failed compile, Spike run
     or proof goes back to Gemini for a repair; --max-repairs caps the repairs of a kernel at one VLEN, so it
     makes at most 1 + --max-repairs Gemini calls.
  2  The fully symbolic sweep of artifact/ at the same VLEN, 600 s: every input bit pattern, symbolic
     parameters within XNNPACK's constraints (artifact/elementwise.py, or growth_sweep for the
     multidimensional kernels). Its result is the verdict: a kernel passes when the sweep verifies at least
     size 1 (level 1) and finds no counterexample or memory violation.

Every verifier is one single-threaded solver and its bound is "how far it got in 600 s", so kernels wait
in one queue and at most one runs per physical core (--workers or $SALT_JOBS overrides), pinned to that core
with everything it starts (taskset), so no other kernel's work runs there. The VLENs run one after another. Results go to out/phase1/vlen<V>/; the shipped kernels/target is never written.
Phase 1 runs once per kernel and VLEN: rerunning a command continues where it stopped, does not translate a
kernel whose Phase 1 failed again (no Gemini call is spent twice), and redoes only a sweep that broke;
--fresh starts the chosen kernels over; --resweep redoes their sweeps only (after a verifier change, no Gemini).
--regate runs Phase 1 again on their last translations (after a harness fix; repairs as usual, no new
translation); with --no-repairs it makes no Gemini call at all.

Environment: Gemini as for ae/e2e.sh (GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION with
application-default credentials), CHIPYARD_PATH (default ~/chipyard; Spike), PHASE1_MODEL (default
gemini-3.8-flash; the paper used gemini-3.1-pro-preview-customtools)."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifact"))
import paths          # noqa: E402  kernel lists and the physical core count (also puts REPO on sys.path)
import elementwise    # noqa: E402  the Phase-1 NaN-rule tags and its output layout

SECONDS = 600                     # the paper's budget, for the pipeline's proof and for the sweep
MAX_REPAIRS = 10                  # salty-rn's --max-compile-retries 5 + --max-verification-retries 5
SLOT_BASE = 100                   # build/slots/slot_100.. : apart from slot 0 (salty-rn -k, ae/e2e.sh)
MINUTES_PER_KERNEL = 24           # rough: Phase 1 ~13 min (the paper's average), sweep ~11 min
SWEEP_MINUTES = 11                # the sweep alone (600 s, plus compile and coverage)
SWEEP_FAILURES = ("COUNTEREXAMPLE", "MEMORY", "CANDIDATE_REJECTS", "FIXTURE_ERROR",
                  "VACUOUS_ASSUMPTIONS", "VACUOUS_CONSTANT_OUTPUT", "NATIVE_ACCESS")
GROUPS = {"all": paths.ELEMENTWISE + paths.MULTIDIM, "elementwise": paths.ELEMENTWISE, "multidim": paths.MULTIDIM}
CY = Path(os.environ.get("CHIPYARD_PATH") or Path.home() / "chipyard").expanduser()
SALTY = Path(sys.executable).with_name("salty-rn")

_procs, _procs_lock, _stop, _print_lock = set(), threading.Lock(), threading.Event(), threading.Lock()


class Interrupted(Exception):
    pass


def say(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def family(k):
    return "elementwise" if k in paths.ELEMENTWISE else "multidim"


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def child_env(**extra):
    env = dict(os.environ, CHIPYARD_PATH=str(CY), **extra)
    env.pop("ZEPHYR_BASE", None)          # SALTyRN's own third_party/zephyr, not a shell leftover
    env["PATH"] = f"{REPO / '.venv/bin'}:{CY / '.conda-env/riscv-tools/bin'}:{env.get('PATH', '')}"
    return env


def core_cpus():
    """The logical CPUs of every physical core this process may use (hyperthread siblings together), so a
    kernel can be pinned to a core of its own. [] where the topology is not readable (no pinning)."""
    cores = {}
    try:
        for c in sorted(os.sched_getaffinity(0)):
            t = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
            key = ((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip())
            cores.setdefault(key, []).append(c)
    except (AttributeError, OSError):
        return []
    return sorted(cores.values())


CORES = core_cpus()
TASKSET = shutil.which("taskset")


def run(cmd, log, cwd, env, cpus=None):
    """Run cmd with its output in log, on the CPUs `cpus` (one physical core: cmd and everything it starts
    inherit them); Ctrl-C (kill_all) stops it and everything it started."""
    if _stop.is_set():
        raise Interrupted
    if cpus and TASKSET:
        cmd = [TASKSET, "-c", ",".join(map(str, cpus))] + list(cmd)       # taskset execs cmd: same process
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        with _procs_lock:
            _procs.add(p)
            late = _stop.is_set()          # kill_all already ran: stop this one too
        if late:
            _kill_tree([p.pid])
        try:
            rc = p.wait()
        finally:
            with _procs_lock:
                _procs.discard(p)
    if _stop.is_set():
        raise Interrupted
    return rc


def _descendants(pids):
    """pids and every process below them (Linux /proc). The verifiers start their solvers in new sessions,
    so killing a process group alone would leave them running."""
    children = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                children.setdefault(int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]), []).append(int(d.name))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], list(pids)
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def _kill_tree(top):
    try:
        everything = _descendants(top)
    except OSError:                       # no /proc: the process groups only
        everything = []
    for pid in top:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in everything:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def kill_all():
    with _procs_lock:
        _stop.set()
        top = [p.pid for p in _procs]
    _kill_tree(top)


# ---------------------------------------------------------------- the two steps of one kernel

def phase1(k, vlen, vdir, slot, s, cpus=None, regate=False):
    """salty-rn on one kernel in its own kernels directory and build slot. With `regate`, Phase 1 again on the
    kernel's last translation (after a harness or pipeline fix): no new translation, repairs as usual."""
    work = vdir / "work" / k / "kernels"
    cand = vdir / "rvv" / f"{k}.c"
    log = vdir / "logs" / f"{k}.phase1.log"
    target = work / "target" / f"{k}{'' if vlen == 256 else f'-{vlen}'}.c"
    last = target.read_text() if regate and target.exists() else None
    if regate and last is None:
        raise RuntimeError(f"--regate: no earlier translation of {k} at VLEN {vlen} ({target})")
    shutil.rmtree(work.parent, ignore_errors=True)
    for sub in ("source", "harness"):
        (work / sub).mkdir(parents=True)
        shutil.copy2(REPO / "kernels" / sub / f"{k}.c", work / sub)
    if last is not None:
        target.parent.mkdir(parents=True)
        target.write_text(last)
    cand.unlink(missing_ok=True)
    t0 = time.time()
    repairs = (["--max-repairs", s["max_repairs"]] if s["max_repairs"]          # 0: --regate --no-repairs
               else ["--max-compile-retries", 0, "--max-verification-retries", 0])
    rc = run([SALTY, "-k", k, "--model", s["model"], "--vlen", vlen, "--kernels-dir", work,
              "--verification-timeout", SECONDS, *repairs, "--build-slot", slot,
              "--nan-rule", "sweep"]    # the verdict's NaN rule: no repair is spent on a difference it accepts
             + (["--skip-translation"] if regate else []),
             log, REPO, child_env(), cpus)
    text = log.read_text(errors="replace")
    status = work / "status.json"
    st = json.loads(status.read_text()).get(k, {}) if status.exists() else {}
    err = st.get("error") or ""
    kind = err.split(":", 1)[0] if err else ""
    gate_ran = "Spike: PASS" in text and "Skipping compilation" not in text and "Skipping Spike" not in text
    accepted = bool(rc == 0 and st.get("compiled") and not err and target.exists() and gate_ran)
    error = ("" if accepted else err[:400] if err else f"salty-rn exited with {rc}" if rc
             else "no status.json (see the log)" if not st
             else "the compile + Spike gate did not run" if not gate_ran else "no candidate")
    proofs = re.findall(r"Formal verification: (\w+) \(max_batch=(\d+)\)", text)
    # A multidimensional proof verifies shapes (dims=[..]) and reports max_batch 0: count the last proof's shapes
    last = text.rsplit("Step 3/3: Bounded verification", 1)[-1] if proofs else ""
    shapes = len(re.findall(r"dims=\[[\d,]+\]\s+VERIFIED", last))
    r = dict(accepted=accepted, error=error, seconds=round(time.time() - t0), log=str(log), settings=s,
             attempts=st.get("attempts", 0), compiled=accepted or kind in ("wrong_output", "runtime_error",
                                                                           "formal_verification_counterexample"),
             spike="pass" if accepted or kind == "formal_verification_counterexample"
             else "fail" if kind in ("wrong_output", "runtime_error") else "-",
             proof={"verdict": proofs[-1][0], "max_batch": int(proofs[-1][1]), "shapes": shapes} if proofs else None)
    if accepted:
        cand.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, cand)
        r["rvv_sha256"] = sha256(cand)
    return r


def elementwise_spec(k):
    return next(x for x in elementwise.DEFAULT["1"] if x.partition("@")[0] == k)      # f32-vmax@ref, ...


def elementwise_verdict(d, spec, vlen):
    """The verdict of artifact/elementwise.py run with SALT_AE_OUT=d."""
    f = d / elementwise.out_dir("1", vlen).name / "summary.json"
    e = json.loads(f.read_text()).get(spec) if f.exists() else None
    if not e or e.get("error"):
        return dict(verdict="ERROR", error=str((e or {}).get("error") or "no result (see the log)")[:400], infra=True)
    # Any failure event counts, not only the first event that stopped the sweep
    ev = d / elementwise.out_dir("1", vlen).name / spec / "events.jsonl"
    ev = [json.loads(x) for x in ev.read_text().splitlines() if x.startswith("{")] if ev.exists() else []
    fail = next((x for x in ev if x.get("status") in SWEEP_FAILURES), None)
    stop = fail or (e.get("stopped") or [{}])[0]
    n = e.get("max_verified", 0)
    verdict = fail["status"] if fail else "VERIFIED" if n else "NONE"
    bound = (f"{verdict.lower()} at n={stop.get('batch', '?')}" if fail else f"1..{n}" if n
             else f"none: {str(stop.get('status', 'no event')).lower()} at n={stop.get('batch', 1)}")
    return dict(verdict=verdict, bound=bound, max_verified=n, stopped=stop, nan_rule=e.get("nan_rule"),
                coverage=e.get("coverage"))


def multidim_verdict(d):
    """The verdict of growth_sweep run with --output d."""
    f = d / "summary.json"
    e = (json.loads(f.read_text()) or [None])[0] if f.exists() else None
    if not e or e.get("status") in ("HARNESS_ERROR", None):
        return dict(verdict="ERROR", error=str((e or {}).get("error") or "no result (see the log)")[:400], infra=True)
    level, status = e.get("completed_level", 0), e["status"]
    if status == "ERROR" and not level:          # the verifier could not run the first shape: say why
        msg = [x for x in (e.get("failure") or {}).get("stderr", "").splitlines() if x.strip()]
        return dict(verdict="ERROR", bound="verifier error at level 1", error=(msg or ["see the log"])[-1][:300],
                    infra=True, completed_level=0, verified_shapes=0, status=status)
    verdict = status if status in SWEEP_FAILURES else "VERIFIED" if level else "NONE"
    where = ",".join(f"{a}={b}" for a, b in (e.get("stopped_shape") or {}).items()) or "?"
    bound = (f"{status.lower()} at {where}" if verdict in SWEEP_FAILURES
             else f"level {level} ({e.get('verified_shapes', 0)} shapes)" if level
             else f"none: {status.lower()} at level 1")
    return dict(verdict=verdict, bound=bound, completed_level=level, verified_shapes=e.get("verified_shapes"),
                status=status, region=e.get("complete_region"), probes=e.get("probes"))


def sweep(k, vlen, vdir, cand, cpus=None):
    """The fully symbolic sweep of one candidate, in its own output directory."""
    d = vdir / "sweep" / k
    log = vdir / "logs" / f"{k}.sweep.log"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    if family(k) == "elementwise":
        spec = elementwise_spec(k)
        d.mkdir(parents=True)
        run([sys.executable, REPO / "artifact/elementwise.py", "--phase", "1", "--vlen", vlen, "--seconds", SECONDS,
             "--workers", 1, "--rvv-dir", cand.parent, spec], log, REPO / "artifact", child_env(SALT_AE_OUT=str(d)),
            cpus)
        v = elementwise_verdict(d, spec, vlen)
    else:
        run([sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", vlen, "--seconds",
             SECONDS, "--coverage", "--rvv-dir", cand.parent, "--kernel", k, "--output", d], log, REPO, child_env(),
            cpus)
        v = multidim_verdict(d)
    return dict(v, seconds_budget=SECONDS, rvv_sha256=sha256(cand), log=str(log), output=str(d),
                seconds=round(time.time() - t0))


def save_row(vdir, k, row):
    """Write one kernel's row into results.json, merged under a lock with what other drivers wrote (two
    drivers on different kernels of one VLEN each own their rows). Returns the merged results."""
    with open(vdir / "results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        f = vdir / "results.json"
        state = json.loads(f.read_text()) if f.exists() else {}
        state[k] = row
        tmp = vdir / f"results.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(f)
        return state


def phase1_done(row, s, cand):
    """Phase 1 runs once per kernel and VLEN: an earlier run with these settings either failed (not retried, so
    no Gemini calls are spent twice; --fresh retries it) or left a translation that is still the candidate."""
    p1 = (row or {}).get("phase1") or {}
    return bool(p1.get("settings") == s and (not p1.get("accepted") or p1.get("rvv_sha256") == sha256(cand)))


def sweep_done(row):
    """An earlier sweep checked this very candidate with this budget (a sweep that broke is redone)."""
    p1, sw = (row or {}).get("phase1") or {}, (row or {}).get("sweep") or {}
    return bool(sw and not sw.get("infra") and sw.get("rvv_sha256") == p1.get("rvv_sha256")
                and sw.get("seconds_budget") == SECONDS)


def kernel_run(k, vlen, vdir, prev, slots, s, resweep=False, regate=False):
    """Phase 1 and then the sweep of one kernel, skipping a step already done for the same inputs."""
    slot, cpus = slots.get()                       # a build slot and its worker's own physical core
    try:
        row = dict(prev or {}, kernel=k, family=family(k), vlen=vlen)
        cand = vdir / "rvv" / f"{k}.c"
        if regate or not phase1_done(row, s, cand):
            say(f"VLEN {vlen} {k}: Phase 1{' again on its last translation' if regate else ''} (slot {slot})")
            row.pop("sweep", None)
            row["phase1"] = phase1(k, vlen, vdir, slot, s, cpus, regate)
            if regate:
                row["phase1"]["regated"] = True
        p1 = row["phase1"]
        if not p1["accepted"]:
            row.pop("sweep", None)
        elif resweep or not sweep_done(row):
            say(f"VLEN {vlen} {k}: fully symbolic sweep")
            row["sweep"] = sweep(k, vlen, vdir, cand, cpus)
        row["passed"] = bool(p1["accepted"] and row.get("sweep", {}).get("verdict") == "VERIFIED")
        return row
    finally:
        slots.put((slot, cpus))


# ---------------------------------------------------------------- report

def table(rows, vlen, max_repairs):
    head = ("kernel", "family", "compile", "spike", "pipeline proof", "repairs", "fully symbolic sweep", "time",
            "result")
    lines = []
    for r in rows:
        p1, sw = r.get("phase1") or {}, r.get("sweep") or {}
        proof = p1.get("proof") or {}
        ok = proof.get("verdict") in ("ALL_PASSED", "PARTIAL")
        pr = (f"1..{proof['max_batch']}" if ok and proof.get("max_batch")
              else f"{proof['shapes']} shapes" if ok and proof.get("shapes") else (proof.get("verdict") or "-").lower())
        rep = f"{max(p1.get('attempts', 1) - 1, 0)}/{p1.get('settings', {}).get('max_repairs', max_repairs)}"
        secs = p1.get("seconds", 0) + sw.get("seconds", 0)
        lines.append((r["kernel"], {"elementwise": "elem", "multidim": "multi"}.get(r["family"], r["family"]),
                      "yes" if p1.get("compiled") else "no", p1.get("spike", "-"), pr, rep,
                      sw.get("bound") or (sw.get("verdict", "-").lower() if sw else "-"),
                      f"{round(secs / 60)}m", "PASS" if r.get("passed") else "FAIL"))
    w = [max(len(str(x[i])) for x in [head] + lines) for i in range(len(head))]
    out = [f"VLEN {vlen}"] + ["  ".join(str(x[i]).ljust(w[i]) for i in range(len(head))).rstrip()
                              for x in [head] + lines]
    failed = [r for r in rows if not r.get("passed")]
    out.append(f"{len(rows) - len(failed)}/{len(rows)} verified at VLEN {vlen}" + (":" if failed else "."))
    for r in failed:
        p1, sw = r.get("phase1") or {}, r.get("sweep") or {}
        why = (f"Phase 1: {p1.get('error') or 'not accepted'}" if not p1.get("accepted")
               else f"sweep: {sw.get('verdict')} {sw.get('error') or sw.get('bound') or ''}")
        out.append(f"  FAIL {r['kernel']}: {why.splitlines()[0][:150]}")
        out.append(f"       log: {sw.get('log') if p1.get('accepted') else p1.get('log')}")
    return "\n".join(out)


# ---------------------------------------------------------------- preflight

def gemini_access():
    """The check ae/e2e.sh makes: an API key, or Vertex AI with application-default credentials."""
    import workflow.llm.client as c
    k = c._KEYS
    if k["GOOGLE_API_KEY"] and not k["GOOGLE_CLOUD_PROJECT"]:
        return "API key"
    if not k["GOOGLE_CLOUD_PROJECT"]:
        sys.exit("Gemini: set GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (env or "
                 "src/workflow/llm/keys.py)")
    import google.auth
    try:
        google.auth.default()
    except Exception as e:
        sys.exit(f"Gemini: Vertex project {k['GOOGLE_CLOUD_PROJECT']} but no application-default credentials "
                 f"({type(e).__name__}); run: gcloud auth application-default login --no-launch-browser")
    return f"Vertex AI, project {k['GOOGLE_CLOUD_PROJECT']}, location {k['GOOGLE_CLOUD_LOCATION']}"


def preflight(model, need_phase1):
    """Everything the gate and the proofs need, so that nothing is skipped silently (salty-rn accepts an
    untested translation when it finds no Zephyr or no Spike)."""
    zephyr = REPO / "third_party/zephyr"
    checks = [(SALTY.exists(), f"no {SALTY}: run through ./ae.sh, or install the package in .venv"),
              ((zephyr / "scripts/set_envvars_sdk.sh").exists(), f"no Zephyr at {zephyr} (git submodule)"),
              ((REPO / "build/templates/xnnpack-kernel/app").exists(), "no Zephyr app template under build/templates"),
              ((CY / "env.sh").exists() and (CY / ".conda-env/riscv-tools/bin/spike").exists(),
               f"no Chipyard with Spike at {CY} (set CHIPYARD_PATH)"),
              (paths.CVC5.exists(), f"no cvc5 at {paths.CVC5} (see the README)")]
    bad = [m for ok, m in checks if not ok]
    if bad:
        sys.exit("cannot run Phase 1:\n  " + "\n  ".join(bad))
    if need_phase1 and "gemini" in model:
        return gemini_access()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the 35 kernels by family")
    ap.add_argument("--kernels", help="comma-separated kernel names, or all, elementwise, multidim")
    ap.add_argument("--vlen", default="128,512", help="comma-separated VLENs, run in this order (default 128,512)")
    ap.add_argument("--workers", type=int, default=0, help="kernels at once (default: one per physical core)")
    ap.add_argument("--max-repairs", type=int, default=MAX_REPAIRS,
                    help=f"Gemini repairs per kernel and VLEN (default {MAX_REPAIRS})")
    ap.add_argument("--model", default=os.environ.get("PHASE1_MODEL", "gemini-3.8-flash"))
    ap.add_argument("--out", type=Path, default=REPO / "out/phase1")
    ap.add_argument("--fresh", action="store_true", help="discard earlier results of the chosen kernels")
    ap.add_argument("--regate", action="store_true",
                    help="Phase 1 again on the chosen kernels' last translations, with no new translation "
                         "(after a harness or pipeline fix; repairs as usual), then the sweep")
    ap.add_argument("--no-repairs", action="store_true",
                    help="with --regate: no Gemini call at all, the gate and the proof on the last translation as "
                         "it is (accepted or not)")
    ap.add_argument("--resweep", action="store_true",
                    help="redo the fully symbolic sweep of the chosen kernels and keep their Phase 1 (no Gemini): "
                         "after a verifier change")
    a = ap.parse_args()

    if a.list:
        for name, ks in (("elementwise", paths.ELEMENTWISE), ("multidimensional", paths.MULTIDIM)):
            print(f"{name} ({len(ks)}):")
            for i in range(0, len(ks), 4):
                print("  " + "".join(k.ljust(26) for k in ks[i:i + 4]).rstrip())
        return
    if not a.kernels:
        ap.error("choose kernels with --kernels (names from --list, or all)")
    kernels = []
    for x in a.kernels.split(","):
        x = x.strip()
        if x in GROUPS:
            kernels += GROUPS[x]
        elif x in GROUPS["all"]:
            kernels.append(x)
        elif x:
            ap.error(f"unknown kernel {x!r}; see --list")
    kernels = [k for k in GROUPS["all"] if k in set(kernels)]          # the paper's order, no duplicates
    try:
        vlens = list(dict.fromkeys(int(v) for v in a.vlen.split(",")))
    except ValueError:
        ap.error("--vlen takes numbers, e.g. 128,512")
    if not vlens or any(v < 128 or v & (v - 1) for v in vlens):
        ap.error("every VLEN must be a power of two >= 128")
    if a.fresh + a.resweep + a.regate > 1:
        ap.error("--fresh, --resweep and --regate are alternatives: choose one")
    if a.max_repairs < 1:
        ap.error("--max-repairs must be at least 1")
    if a.no_repairs and not a.regate:
        ap.error("--no-repairs goes with --regate (a new translation is a Gemini call)")
    workers = min(a.workers or paths.physical_cores(), len(kernels))
    s = dict(model=a.model, max_repairs=0 if a.no_repairs else a.max_repairs, seconds=SECONDS)
    out = a.out.resolve()

    # What is left to do, from earlier runs
    states, todo1, todo, minutes = {}, 0, 0, 0
    for v in vlens:
        f = out / f"vlen{v}" / "results.json"
        states[v] = json.loads(f.read_text()) if f.exists() else {}
        left, work = 0, 0
        for k in kernels:
            if a.fresh:
                states[v].pop(k, None)
            r = states[v].get(k)
            p1 = phase1_done(r, s, out / f"vlen{v}" / "rvv" / f"{k}.c")
            need_sweep = not (p1 and sweep_done(r) and not a.resweep)
            p1 = p1 and not a.regate
            need_sweep = need_sweep or a.regate
            todo1 += not p1
            left += need_sweep
            work += MINUTES_PER_KERNEL if not p1 else SWEEP_MINUTES if need_sweep else 0
        todo += left
        minutes += math.ceil(work / workers) if work else 0
    gemini = preflight(a.model, todo1 > 0 and not a.no_repairs)
    calls = 0 if a.no_repairs else todo1 * (a.max_repairs + (not a.regate))     # --regate: no new translation
    print(f"Phase 1: {len(kernels)} kernels x VLEN {', '.join(map(str, vlens))}, {workers} at once "
          f"({'each pinned to its own physical core' if CORES and TASKSET and workers <= len(CORES) else 'one per physical core, not pinned'}) -> {out}")
    print(f"  model {a.model}{f' ({gemini})' if gemini else ''}; at most {s['max_repairs']} repairs per kernel and "
          f"VLEN: {todo1} runs need Phase 1, at most {calls} Gemini calls")
    print(f"  {SECONDS} s pipeline proof + {SECONDS} s fully symbolic sweep per kernel; {todo} of "
          f"{len(kernels) * len(vlens)} runs to do, roughly {minutes / 60:.1f} h (about {MINUTES_PER_KERNEL} min "
          f"per kernel, {SWEEP_MINUTES} for a sweep alone, more when a kernel needs verification repairs)", flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):   # SIGINT too: a script's background job ignores it
        signal.signal(sig, interrupt)
    slots = queue.Queue()
    for i in range(workers):                 # a slot per worker, named after its core: drivers on other cores
        cpus = CORES[i % len(CORES)] if CORES else None     # never share a build slot
        slots.put((SLOT_BASE + (cpus[0] if cpus else i), cpus))
    # Longest first, so that the last core is not left with a long kernel: multidimensional, then elementwise.
    order = [k for k in kernels if family(k) == "multidim"] + [k for k in kernels if family(k) == "elementwise"]
    reports, all_passed = [], True
    for v in vlens:
        vdir = out / f"vlen{v}"
        vdir.mkdir(parents=True, exist_ok=True)
        state = states[v]
        say(f"VLEN {v}: {len(kernels)} kernels, {workers} at once")
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(kernel_run, k, v, vdir, state.get(k), slots, s, a.resweep, a.regate): k
                   for k in order}
        done = 0
        try:
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    row = fut.result()
                except Interrupted:
                    continue
                except Exception as e:                 # one kernel's crash must not stop the others
                    row = dict(kernel=k, family=family(k), vlen=v, passed=False,      # settings: not retried either
                               phase1=dict(accepted=False, settings=s, error=f"driver: {type(e).__name__}: {e}"[:400]))
                state[k] = row
                state = save_row(vdir, k, row)
                done += 1
                p1, sw = row.get("phase1") or {}, row.get("sweep") or {}
                say(f"VLEN {v} {k}: {'PASS' if row.get('passed') else 'FAIL'}  Phase 1 "
                    f"{'accepted' if p1.get('accepted') else 'failed'} ({max(p1.get('attempts', 1) - 1, 0)} repairs)"
                    + (f", sweep {sw.get('verdict')} {sw.get('bound', '')}" if sw else "") + f"  [{done}/{len(kernels)}]")
        except KeyboardInterrupt:
            say("interrupted: stopping every kernel; finished kernels are kept, rerun the command to continue")
            kill_all()
            pool.shutdown(wait=True, cancel_futures=True)
            sys.exit(130)
        pool.shutdown(wait=True)
        rows = [state[k] for k in kernels if k in state]
        rep = table(rows, v, a.max_repairs)
        (vdir / "summary.txt").write_text(rep + "\n")
        reports.append(rep)
        all_passed &= all(r.get("passed") for r in rows)
        print("\n" + rep + "\n", flush=True)
    if len(vlens) > 1:
        print("\n\n".join(reports))
    print(f"\nresults: {', '.join(str(out / f'vlen{v}' / 'results.json') for v in vlens)}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
