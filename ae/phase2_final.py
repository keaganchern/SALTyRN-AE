#!/usr/bin/env python3
"""Phase 2's final verification, ae/e2e.sh step 3 (the paper's "Candidate Evaluation and Final Verification").

1. The search's kept candidates (each passed Autocomp's Spike check) are proved against the verified Phase-1 kernel,
   fastest first; the fastest that verifies is the result. A Spike check is not a proof: the fastest candidate can be
   wrong on inputs the harness never uses.
2. A finalist that failed and is faster than that result is repaired with the counterexample (LLM), at most
   --max-repairs times each and --repair-budget times in all. A repaired candidate re-enters the flow: compile and
   Spike check against Phase 1 (Autocomp's harness), FireSim timing, then the proof; it becomes the result when it
   verifies and is faster.
3. If no candidate verifies, the Phase-1 kernel stays the result.

Writes <out>/phase2/<kernel>.c (the result), <out>/verify/phase2/<kernel>/ (its proof) and <out>/phase2/final.json
(every proof and repair). State is kept in final.json, so a run can resume; exit code 75 means a repaired candidate
needs FireSim timing and no run farm is up (--no-firesim): e2e.sh launches one and runs this again.
"""
import argparse, glob, json, os, pathlib, re, shutil, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
AC = REPO / "third_party/autocomp"
OK = ("VERIFIED", "TIMEOUT", "MAX_LEVEL_REACHED")
NEED_FIRESIM = 75
Q3 = "'" * 3


def norm(c):
    return re.sub(r"\s+", " ", re.sub(r"/\*.*?\*/|//[^\n]*", "", c, flags=re.S)).strip()


def entry_test_rvv(src):
    """The entry point named test_rvv, like every artifact kernel (helpers such as `static void pack_weights` first)."""
    names = re.findall(r"^\s*void\s+(\w+)\s*\(", src, re.M) or re.findall(r"\bvoid\s+(\w+)\s*\(", src)
    if names and "test_rvv" not in names:
        src = re.sub(r"(^\s*void\s+)" + re.escape(names[-1]) + r"(\s*\()", r"\1test_rvv\2", src, count=1, flags=re.M)
    return src


def ranked_candidates(sd, maxc):
    own_re = re.compile("code=" + Q3 + "(.*?)" + Q3 + r"\s*,\s*score=([0-9.]+|None)", re.S)
    base, cands = "", {}
    for f in sorted(glob.glob(f"{sd}/candidates-iter-*/candidate_*.txt")):
        own = list(own_re.finditer(open(f, errors="replace").read()))   # ancestors first: own code and score last
        if not own:
            continue
        code, score = own[-1].group(1), own[-1].group(2)
        it = int(re.search(r"candidates-iter-(\d+)", f).group(1))
        if it == 0:
            base = code
            continue
        if score != "None" and (norm(code) not in cands or float(score) < cands[norm(code)][0]):
            cands[norm(code)] = (float(score), it, code)
    ranked = sorted(v for k, v in cands.items() if k != norm(base))[:maxc]
    return [{"name": f"rank{r}", "score": s, "iter": it, "code": entry_test_rvv(c)} for r, (s, it, c) in enumerate(ranked, 1)], len(cands)


def baseline_score(sd):
    f = glob.glob(f"{sd}/eval-results-iter-0/code_0_result.txt")
    try:
        return json.load(open(f[0])).get("score") if f else None
    except (OSError, ValueError):
        return None


def prove(a, code, name):
    """The Phase-2 proof of `code` against <out>/phase1, as ae/verify.py runs it; returns (outcome, proof dir)."""
    cdir = a.out / "phase2" / name
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{a.kernel}.c").write_text(code)
    o = a.out / "verify" / f"phase2-{name}" / a.kernel
    shutil.rmtree(o, ignore_errors=True)
    o.mkdir(parents=True)
    env = dict(os.environ, SALT_AE_OUT=str(o), SALT_CEX_JSON=str(o), SALT_PHASE1_DIR=str(a.out / "phase1"))
    if a.family == "elementwise":
        cmd = [sys.executable, "elementwise.py", "--phase", "2", "--vlen", str(a.vlen), "--seconds", str(a.seconds),
               "--workers", "1", "--rvv-dir", str(cdir), *a.specs2.split(",")]
        cwd = REPO / "artifact"
    else:
        cmd = [sys.executable, "-m", "src.workflow.verification.engine.growth_sweep", "--vlen", str(a.vlen), "--seconds",
               str(a.seconds), "--coverage", "--rvv-dir", str(cdir), "--kernel", a.kernel, "--output", str(o / "sweep"),
               "--reference-dir", str(a.out / "phase1")]
        cwd = REPO
    with open(a.out / f"3-verify-phase2-{name}.log", "w") as log:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return outcome(o), o


def outcome(d):
    st = []
    for f in glob.glob(f"{d}/elementwise_phase*/summary.json"):
        for r in json.load(open(f)).values():
            s = "ERROR" if r.get("stage") or r.get("error") else ((r.get("stopped") or [{}])[0].get("status") or "VERIFIED")
            st.append(s if r.get("max_verified") or s not in ("VERIFIED", "TIMEOUT") else "NO_SIZE_VERIFIED")
    for f in glob.glob(f"{d}/sweep/summary.json"):
        for r in json.load(open(f)):
            s = r.get("status", "ERROR")
            st.append(s if r.get("completed_level") or s not in OK else "NO_SIZE_VERIFIED")
    bad = [s for s in st if s not in OK]
    return bad[0] if bad else (st[0] if st else "NO_RESULT")


def counterexample_text(d, src=""):
    """What the repair prompt shows: where the outputs differ, and the counterexample's inputs (from the dump); for
    a refusal (src: the candidate's code), what the verifier could not follow, where, and what to change."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.workflow.verification import refusal
    lines = []
    refused = lambda why: lines.append(refusal.describe(why.strip().rstrip("."), src))
    for f in glob.glob(f"{d}/elementwise_phase*/summary.json"):
        for spec, r in json.load(open(f)).items():
            err = " ".join(r.get("error")) if isinstance(r.get("error"), list) else str(r.get("error") or "")
            if "candidate: " in err:
                refused(err.split("candidate: ", 1)[-1])
            elif refusal.missing_intrinsics(err):
                refused("the verifier has no model for " + ", ".join(refusal.missing_intrinsics(err)))
            for s in r.get("stopped") or []:
                if s.get("status") == "NATIVE_ACCESS":
                    refused(str(r.get("stderr_tail", "")).split("NATIVE_ACCESS:", 1)[-1])
                elif s.get("status") not in OK:
                    lines.append(f"{spec}: {s.get('status')} at batch {s.get('batch')} (first differing output element "
                                 f"{s.get('fail_index', '?')})")
    for f in glob.glob(f"{d}/sweep/summary.json"):
        for r in json.load(open(f)):
            for e in (r.get("failure") or {}).get("events", []):
                if e.get("status") not in ("SOLVING", "NONVACUOUS", "VERIFIED"):
                    lines.append(", ".join(f"{k}={v}" for k, v in e.items() if k not in ("config",)))
            err = str(r.get("error") or "")
            if "candidate: " in err:
                refused(err.split("candidate: ", 1)[-1])
            elif refusal.missing_intrinsics(err):
                refused("the verifier has no model for " + ", ".join(refusal.missing_intrinsics(err)))
            # why a native access was refused (a C store, native float arithmetic on an input...): the harness's stderr
            for l in [l for l in ((r.get("failure") or {}).get("stderr") or "").splitlines() if l.startswith("NATIVE_ACCESS:")][:1]:
                refused(l.split("NATIVE_ACCESS:", 1)[1])
    for f in sorted(glob.glob(f"{d}/cex-*.json"))[:1]:
        cex = json.load(open(f))
        lines.append(f"inputs at shape {cex['shape']} (little-endian bytes, first 64 per buffer):")
        for name, hexs in cex["buffers"].items():
            if name.endswith("_rvv"):
                continue
            lines.append(f"  {name.replace('_neon', ' (reference output)')}: {hexs[:128]}")
            if name.endswith("_neon"):
                rvv = cex["buffers"].get(name[:-5] + "_rvv", "")
                lines.append(f"  {name[:-5]} (optimized output): {rvv[:128]}")
    return "\n".join(lines) or "the proof found a difference (no details recorded)"


def autocomp_eval(a, code, simulator):
    """Autocomp's own evaluation of one candidate on its problem (compile, Spike check against Phase 1; with
    simulator=firesim also the FireSim timing). Returns the stats dict (correct, latency, score...)."""
    py = ("import json, sys\n"
          "from autocomp.backend.saturn.saturn_eval import SaturnEvalBackend\n"
          "from autocomp.search.prob import Prob\n"
          f"b = SaturnEvalBackend(vlen={a.vlen})\n"
          "code = open(sys.argv[1]).read()\n"
          f"r = b.evaluate_code(Prob({a.prob_type!r}, {int(a.prob_id)}), [code], {simulator!r})[0]\n"
          "print('AUTOCOMP_RESULT ' + json.dumps(r, default=str))\n")
    src = a.out / "phase2" / "repair_eval.c"
    src.write_text(code)
    env = dict(os.environ, PYTHONPATH=str(AC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", py, str(src)], cwd=AC, env=env, capture_output=True, text=True)
    with open(a.out / "3-repair-autocomp.log", "a") as log:
        log.write(r.stdout[-20000:] + r.stderr[-20000:])
    m = re.findall(r"^AUTOCOMP_RESULT (.*)$", r.stdout, re.M)
    return json.loads(m[-1]) if m else {"correct": False, "error": (r.stderr or r.stdout)[-600:]}


def repair_with_llm(a, ref_code, cand_code, cex):
    sys.path.insert(0, str(REPO / "src"))
    from workflow.llm.client import LLMClient          # noqa: E402
    from workflow.pipeline import extract_code         # noqa: E402
    sig = re.search(r"void\s+test_rvv\s*\([^)]*\)", cand_code, re.S)
    prompt = ((REPO / "prompts/Phase2/verification_repair.md").read_text()
              .replace("{reference_code}", ref_code).replace("{candidate_code}", cand_code)
              .replace("{counterexample}", cex).replace("{signature}", re.sub(r"\s+", " ", sig.group(0)) if sig else "void test_rvv(...)"))
    reply = LLMClient(a.model).chat(prompt, temperature=1.0, thinking="medium")
    return entry_test_rvv(extract_code(reply) or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--kernel", required=True); ap.add_argument("--vlen", type=int, required=True)
    ap.add_argument("--family", required=True); ap.add_argument("--specs2", required=True)
    ap.add_argument("--search-dir", required=True); ap.add_argument("--prob-type", required=True)
    ap.add_argument("--prob-id", required=True); ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--max-candidates", type=int, default=8); ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--repair-budget", type=int, default=4); ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--no-firesim", action="store_true", help="no run farm is up: exit 75 when timing is needed")
    a = ap.parse_args()
    a.out = a.out.resolve()
    statef = a.out / "phase2" / "final.json"
    st = json.load(open(statef)) if statef.exists() else {}
    save = lambda: statef.write_text(json.dumps(st, indent=1))
    (a.out / "phase2").mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"    {m}", flush=True)

    if "candidates" not in st:
        cands, n = ranked_candidates(a.search_dir, a.max_candidates)
        st.update(candidates=cands, distinct=n, baseline_score=baseline_score(a.search_dir), proofs={}, repairs=[],
                  chosen=None, repair_calls=0)
        log(f"{n} distinct candidates kept by the search; proving up to the {len(cands)} fastest")
        save()
    ref_code = (a.out / "phase1" / f"{a.kernel}.c").read_text()

    # 1. Fastest first until one verifies.
    if st["chosen"] is None:
        for c in st["candidates"]:
            if c["name"] in st["proofs"]:
                res = st["proofs"][c["name"]]
            else:
                res, _ = prove(a, c["code"], c["name"])
                st["proofs"][c["name"]] = res
                save()
                log(f"candidate {c['name']} (score {c['score']:.0f}): {res}")
            if res in OK:
                st["chosen"] = {"name": c["name"], "score": c["score"], "kind": "candidate"}
                break
        if st["chosen"] is None:
            st["chosen"] = {"name": "phase1", "score": st["baseline_score"], "kind": "phase1"}
        save()

    # 2. Repair the failed finalists faster than the result, fastest first.
    def beats(score):
        cur = st["chosen"]["score"]
        return score is not None and (cur is None or score < cur)
    for c in st["candidates"]:
        if st["proofs"].get(c["name"]) in OK or not beats(c["score"]):
            continue
        done = [r for r in st["repairs"] if r["of"] == c["name"]]
        code = done[-1]["code"] if done and done[-1].get("code") else c["code"]
        cex_dir = a.out / "verify" / (f"phase2-{done[-1]['name']}" if done and done[-1].get("proof") else f"phase2-{c['name']}") / a.kernel
        while True:
            pend = [r for r in st["repairs"] if r["of"] == c["name"] and r.get("stage") == "needs FireSim"]
            if not pend and (len([r for r in st["repairs"] if r["of"] == c["name"]]) >= a.max_repairs
                             or st["repair_calls"] >= a.repair_budget):
                break
            if pend:
                rep = pend[-1]
            else:
                k = len([r for r in st["repairs"] if r["of"] == c["name"]]) + 1
                rep = {"of": c["name"], "name": f"{c['name']}-repair{k}"}
                st["repair_calls"] += 1
                try:
                    rep["code"] = repair_with_llm(a, ref_code, code, counterexample_text(cex_dir, code))
                except Exception as e:                    # noqa: BLE001
                    rep.update(stage="LLM error", error=str(e)[:300])
                    st["repairs"].append(rep); save(); log(f"{rep['name']}: LLM error {e}")
                    break
                if not re.search(r"\bvoid\s+test_rvv\s*\(", rep["code"]):     # no kernel in the reply (a snippet)
                    rep.update(stage="no code in the reply"); st["repairs"].append(rep); save(); continue
                spike = autocomp_eval(a, rep["code"], "spike")
                if not spike.get("correct"):
                    rep.update(stage="Spike check failed", detail=str(spike.get("error") or spike)[:300])
                    st["repairs"].append(rep); save(); log(f"{rep['name']}: {rep['stage']}")
                    code = rep["code"]
                    continue
                st["repairs"].append(rep)
            if a.no_firesim:
                rep["stage"] = "needs FireSim"
                save()
                log(f"{rep['name']}: passed the Spike check; FireSim timing needed")
                sys.exit(NEED_FIRESIM)
            fs = autocomp_eval(a, rep["code"], "firesim")
            rep.update(score=fs.get("score"), latency=fs.get("latency"))
            if not fs.get("correct") or fs.get("score") is None:
                rep["stage"] = "FireSim run failed"; save(); log(f"{rep['name']}: {rep['stage']}"); break
            if not beats(fs["score"]):
                rep["stage"] = "not faster than the result"; save(); log(f"{rep['name']}: score {fs['score']}, not faster"); break
            res, _ = prove(a, rep["code"], rep["name"])
            rep.update(proof=res, stage="verified" if res in OK else "proof failed")
            save()
            log(f"{rep['name']}: score {fs['score']}, proof {res}")
            if res in OK:
                st["chosen"] = {"name": rep["name"], "score": fs["score"], "kind": "repaired"}
                save()
                break
            code, cex_dir = rep["code"], a.out / "verify" / f"phase2-{rep['name']}" / a.kernel

    # 3. The result.
    ch = st["chosen"]
    if ch["kind"] == "phase1":
        shutil.copy(a.out / "phase1" / f"{a.kernel}.c", a.out / "phase2" / f"{a.kernel}.c")
        log("no candidate verified: the Phase-1 kernel stays the result (speedup 1.0)")
    else:
        shutil.copy(a.out / "phase2" / ch["name"] / f"{a.kernel}.c", a.out / "phase2" / f"{a.kernel}.c")
        dst = a.out / "verify" / "phase2" / a.kernel
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(a.out / "verify" / f"phase2-{ch['name']}" / a.kernel, dst)
        log(f"the Phase-2 kernel: {ch['name']} ({ch['kind']}, score {ch['score']})")
    (a.out / "phase2" / "chosen").write_text(ch["name"])
    save()


if __name__ == "__main__":
    main()
