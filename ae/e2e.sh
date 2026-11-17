#!/usr/bin/env bash
# SALTyRN end to end on one kernel, at VLEN 512 (Saturn VLEN 512 / DLEN 256) or 128 (DLEN 64):
#
#   1  Phase 1   translate Neon -> RVV (Gemini), compile + Spike gate, SMT: Neon vs Phase-1 RVV
#   2  Phase 2   Autocomp beam search, candidates checked on Spike and timed on FireSim (one AWS f2.6xlarge)
#   3  verify    SMT, fully symbolic, growing sizes, the same proofs as ./ae.sh verify (family and tags from
#                kernels/phase<P>/vlen<V>/manifest.json): Neon vs Phase-1 RVV, and Phase-1 RVV vs the search's
#                finalists, fastest first, until one verifies; faster ones that fail are repaired with their
#                counterexample, re-checked on Spike, timed on FireSim and proved (ae/phase2_final.py)
#   4  report    cycles of the Phase-1 baseline and the optimized kernel, speedup, verified bounds
#
#   ae/e2e.sh <kernel> [--vlen 128|512]              default f32-vadd --vlen 512; out/e2e-<kernel>-<vlen>/
#   ae/e2e.sh <kernel> <autocomp prob id>            (older form, VLEN 512)
# Any of the 35 kernels. The Autocomp problem is found from the kernel name (sols/cgo-kernels/ at VLEN 512,
# sols/cgo-kernels-128/ at 128); x32-transposec has none at VLEN 512, so there it runs Phase 1 only (STOP=1).
#
# Environment
#   Gemini (steps 1, 2)  GOOGLE_API_KEY, or Vertex AI: GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (env or the two
#                       keys.py files) with application-default credentials (gcloud auth application-default login)
#   AWS credentials     for the FireSim manager (step 2); the account is printed before anything launches
#   CHIPYARD_PATH       Chipyard with FireSim set up (default ~/chipyard), with the Saturn FPGA images in
#                       config_hwdb.yaml and workload saturn.json (./ae.sh setup); step 2 switches
#                       config_runtime.yaml to the image of the VLEN and puts the previous image back afterwards
#   ITERS               Autocomp iterations (default 8, the paper's T); 1 for a quick check
#   MAX_CANDIDATES      step 3 proves the search's fastest candidates up to this many (default 8)
#   MAX_REPAIRS         repairs of a failed finalist faster than the result (default 2); REPAIR_BUDGET in all
#                       (default 4); REPAIR_MODEL (default PHASE1_MODEL)
#   PHASE1_MODEL        Phase 1 translation model (default gemini-3.8-flash; the paper used
#                       gemini-3.1-pro-preview-customtools)
#   AUTOCOMP_MODELS     Phase 2 models, comma-separated (default gemini-3.8-flash,gemini-3.1-pro-preview;
#                       the paper's searches used gemini-3-flash-preview,gemini-3.1-pro-preview)
#   VERIFY_SECONDS      SMT budget per verification (default 600, the paper's)
#   E2E_OUT, START      resume an earlier run: E2E_OUT=<its directory> START=<step>
#   STOP=<step>         stop after that step (STOP=1: Phase 1 only, nothing on AWS)
#   YES=1               do not ask before launching the f2 instance
#   WANDB_MODE          Weights & Biases logging of the search (default disabled: no W&B account needed)
#
# Cost: steps 2-3 run one f2.6xlarge (about $2/hour; roughly 2.5-3 hours at ITERS=8) plus the LLM calls.
# The instance is terminated when step 3 ends, or anything fails or is interrupted.
set -euo pipefail

K=f32-vadd
PID=""
VLEN=512
pos=()
while [ $# -gt 0 ]; do
  case "$1" in
    --vlen) VLEN=${2:?--vlen needs 128 or 512}; shift 2 ;;
    --vlen=*) VLEN=${1#--vlen=}; shift ;;
    -h|--help) sed -n '2,39p' "$0"; exit 0 ;;
    *) pos+=("$1"); shift ;;
  esac
done
[ ${#pos[@]} -ge 1 ] && K=${pos[0]}
[ ${#pos[@]} -ge 2 ] && PID=${pos[1]}
case "$VLEN" in
  512) DLEN=256; BASE=cgo-kernels;     SUF="" ;;
  128) DLEN=64;  BASE=cgo-kernels-128; SUF="-128" ;;
  *) echo "ERROR: --vlen must be 128 or 512, not $VLEN" >&2; exit 1 ;;
esac
HW=firesim_saturn_GENV${VLEN}D${DLEN}_shuttle_singlecore_no_nic_l2_llc4mb_ddr3   # FPGA image of this VLEN
REPO=$(cd "$(dirname "$0")/.." && pwd)
AC=$REPO/third_party/autocomp
CY=${CHIPYARD_PATH:-$HOME/chipyard}
FS=$CY/sims/firesim
ITERS=${ITERS:-8}
VS=${VERIFY_SECONDS:-600}
START=${START:-1}
STOP=${STOP:-4}
OUT=${E2E_OUT:-$REPO/out/e2e-$K-$VLEN}
PROB=e2e-$VLEN                                       # Autocomp problem type holding this run's baseline
ISA=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zvl${VLEN}b_zicntr

export PATH="$REPO/.venv/bin:$CY/.conda-env/riscv-tools/bin:$PATH" CHIPYARD_PATH="$CY"
unset ZEPHYR_BASE                                    # SALTyRN's own third_party/zephyr, not a shell leftover
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/e2e.log"; }
die() { log "ERROR: $*"; exit 1; }
# Run a command in the FireSim manager environment. Chipyard's activation scripts are not written for
# `set -euo pipefail` (e.g. an unset $RISCV), so source them without it; their output goes to the log.
# `source` would hand this function's arguments to sourceme-manager.sh as its options: clear them first.
fs() {
  local cmd=("$@")
  ( set +euo pipefail
    set --
    export PATH="${PATH//$REPO\/.venv\/bin:/}"   # FireSim runs on Chipyard's conda Python, not SALTyRN's venv
    cd "$FS" && source sourceme-manager.sh >> "$OUT/2-firesim.log" 2>&1
    command -v firesim > /dev/null || { echo "FireSim manager environment did not load" >&2; exit 1; }
    "${cmd[@]}" )
}

# ---------------------------------------------------------------- preflight
[ -f "$REPO/kernels/source/$K.c" ] || die "no Neon kernel kernels/source/$K.c"
# The kernel's proofs, as ./ae.sh verify runs them: its family (elementwise or multidim) and the tagged specs of
# each phase (f32-vmax@ref: XNNPACK's NaN rule; qu8-vadd-minmax@multeq,@multne: two parameter cases).
meta=$(python - "$REPO" "$K" "$VLEN" <<'EOF'
import json, sys
repo, k, v = sys.argv[1:]
m = [json.load(open(f"{repo}/kernels/phase{p}/vlen{v}/manifest.json"))["kernels"].get(k) for p in (1, 2)]
if not m[0]:
    sys.exit(f"{k} is not one of the 35 kernels (kernels/phase1/vlen{v}/manifest.json)")
print(m[0]["family"], ",".join(m[0]["specs"]), ",".join((m[1] or m[0])["specs"]))
EOF
) || die "no proofs for $K at VLEN $VLEN"
read -r FAMILY SPECS1 SPECS2 <<< "$meta"
# The Autocomp problem of the kernel, needed for steps 2 and 3: sols/<base>/<id>_<name><suffix>.c (a few sols use another
# name at VLEN 512)
if [ "$START" -le 3 ] && [ "$STOP" -ge 2 ]; then
  if [ -z "$PID" ]; then
    for name in "$K" "${K#f32-}"; do
      f=$(ls "$AC/sols/$BASE/"[0-9]*"_$name$SUF.c" 2>/dev/null | head -1 || true)
      [ -n "$f" ] && { PID=$(basename "$f" | cut -d_ -f1); break; }
    done
    [ -n "$PID" ] || die "no Autocomp problem for $K at VLEN $VLEN (no sols/$BASE/<id>_$K$SUF.c), so no Phase 2; STOP=1 runs Phase 1 alone"
  fi
  ls "$AC/sols/$BASE/${PID}_"*.c > /dev/null 2>&1 || die "no Autocomp problem $PID in sols/$BASE"
  [ -f "$AC/harnesses/$BASE/test$PID.c" ] || die "no Autocomp harness harnesses/$BASE/test$PID.c"
fi
if [ "$START" -le 2 ]; then   # Gemini: an API key, or Vertex AI (project + application-default credentials)
  python - <<'EOF' 2>&1 | tee -a "$OUT/e2e.log" || die "no Gemini access (see above)"
import sys
import workflow.llm.client as c
k = c._KEYS
if k["GOOGLE_API_KEY"] and not k["GOOGLE_CLOUD_PROJECT"]:
    print("    Gemini: API key")
    sys.exit(0)
if not k["GOOGLE_CLOUD_PROJECT"]:
    sys.exit("    Gemini: set GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (env or "
             "src/workflow/llm/keys.py and third_party/autocomp/autocomp/common/keys.py)")
import google.auth
try:
    creds, _ = google.auth.default()
except Exception as e:
    sys.exit(f"    Gemini: Vertex project {k['GOOGLE_CLOUD_PROJECT']} but no application-default credentials "
             f"({type(e).__name__}); run: gcloud auth application-default login --no-launch-browser")
print(f"    Gemini: Vertex AI, project {k['GOOGLE_CLOUD_PROJECT']}, location {k['GOOGLE_CLOUD_LOCATION']}")
EOF
fi
log "kernel $K ($FAMILY${PID:+, Autocomp prob $BASE/$PID}), VLEN $VLEN / DLEN $DLEN, $ITERS iterations, ${VS}s per proof -> $OUT"

# ---------------------------------------------------------------- 1. Phase 1
if [ "$START" -le 1 ]; then
  log "1/4 Phase 1: translate, compile + Spike gate, SMT Neon vs RVV"
  rm -rf "$OUT/kernels" "$OUT/phase1"
  mkdir -p "$OUT/kernels" "$OUT/phase1"
  cp -r "$REPO/kernels/source" "$REPO/kernels/harness" "$OUT/kernels/"   # shipped kernels/target is untouched
  (cd "$REPO" && salty-rn -k "$K" --model "${PHASE1_MODEL:-gemini-3.8-flash}" --vlen "$VLEN" --kernels-dir "$OUT/kernels" --verification-timeout "$VS") \
    > "$OUT/1-phase1.log" 2>&1 || die "Phase 1 failed, see $OUT/1-phase1.log"
  python - "$OUT/kernels/status.json" "$K" <<'EOF' || die "Phase 1 did not produce a verified kernel (see 1-phase1.log)"
import json, sys
s = json.load(open(sys.argv[1])).get(sys.argv[2], {})
print({k: s.get(k) for k in ("generated", "compiled", "verified", "max_verified_batch", "attempts")})
sys.exit(0 if s.get("compiled") and s.get("verified") else 1)
EOF
  cp "$OUT/kernels/target/$K-$VLEN.c" "$OUT/phase1/$K.c"
  log "    Phase-1 RVV: $OUT/phase1/$K.c"
fi
[ "$STOP" -le 1 ] && { log "stopping after step 1 (STOP=1); continue with: E2E_OUT=$OUT START=2 $0 $K --vlen $VLEN"; exit 0; }

# ---------------------------------------------------------------- 2. Phase 2 search on FireSim
LAUNCHED=0
PREV_HW=""
restore_hw() {   # put back the FPGA image config_runtime.yaml had before this run
  if [ -n "$PREV_HW" ]; then
    "$REPO/ae/fs_runtime.sh" set "$FS/deploy" "$PREV_HW" && log "    config_runtime.yaml: default_hw_config back to $PREV_HW" \
      || log "    WARNING: could not restore default_hw_config $PREV_HW in $FS/deploy/config_runtime.yaml"
    PREV_HW=""
  fi
}
teardown() {
  if [ "$LAUNCHED" = 1 ]; then
    log "    terminating the f2 run farm"
    fs firesim terminaterunfarm --forceterminate >> "$OUT/2-firesim.log" 2>&1 \
      || log "    WARNING: terminaterunfarm failed; terminate the f2 instance in the EC2 console"
    LAUNCHED=0
  fi
}
trap 'teardown; restore_hw' EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

stage_problem() {   # this run's Autocomp problem: its Phase-1 kernel as the baseline, the paper's harness
  rm -rf "$AC/sols/$PROB" "$AC/harnesses/$PROB"
  mkdir -p "$AC/sols/$PROB" "$AC/harnesses/$PROB"
  cp "$OUT/phase1/$K.c" "$AC/sols/$PROB/${PID}_$K.c"
  cp "$AC/harnesses/$BASE/test$PID.c" "$AC/harnesses/$PROB/"
  cp "$AC/harnesses/$BASE/context$PID".* "$AC/harnesses/$PROB/" 2>/dev/null || true
}
ensure_firesim() {   # ensure_firesim <what for>: the FPGA image of this VLEN selected and one f2.6xlarge launched
  [ "$LAUNCHED" = 1 ] && return 0
  "$REPO/ae/fs_runtime.sh" check "$FS/deploy" "$HW" \
    || die "FireSim is not configured for $HW (config_hwdb.yaml entry, workload saturn.json): run ./ae.sh setup"
  local cur; cur=$("$REPO/ae/fs_runtime.sh" get "$FS/deploy")
  if [ "$cur" != "$HW" ]; then
    PREV_HW=$cur
    "$REPO/ae/fs_runtime.sh" set "$FS/deploy" "$HW" || die "could not set default_hw_config to $HW"
  fi
  log "    FireSim image $HW ($(grep -A1 "^$HW:" "$FS/deploy/config_hwdb.yaml" | sed -n 's/.*agfi: *//p')), workload saturn.json"
  local acct; acct=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || die "no AWS credentials"
  log "    AWS account $acct; launching 1 x f2.6xlarge (about \$2/hour) for $1"
  if [ "${YES:-0}" != 1 ]; then
    read -r -p "    continue? [y/N] " a; [ "$a" = y ] || [ "$a" = Y ] || die "not confirmed"
  fi
  LAUNCHED=1
  fs firesim launchrunfarm >> "$OUT/2-firesim.log" 2>&1 || die "launchrunfarm failed, see 2-firesim.log"
  # sshd answers before the instance accepts logins; Autocomp's first infrasetup would abort on that
  "$REPO/ae/fs_wait_ssh.sh" "$FS/deploy" >> "$OUT/2-firesim.log" 2>&1 \
    || die "the f2 instance did not accept SSH sessions, see 2-firesim.log"
  log "    run farm up, SSH ready"
}
autocomp_env=(SATURN_CHIPYARD_PATH="$CY" SATURN_SPIKE_ISA="$ISA" AUTOCOMP_PROB_TYPE="$PROB" AUTOCOMP_PROB_ID="$PID"
              AUTOCOMP_VLEN="$VLEN" AUTOCOMP_DLEN="$DLEN" AUTOCOMP_ITERS="$ITERS" WANDB_MODE="${WANDB_MODE:-disabled}"
              PYTHONPATH="$AC${PYTHONPATH:+:$PYTHONPATH}")

if [ "$START" -le 2 ]; then
  log "2/4 Phase 2: Autocomp beam search timed on FireSim"
  [ -f "$OUT/phase1/$K.c" ] || die "no Phase-1 kernel in $OUT/phase1 (run step 1)"
  # The instance stays up through step 3, which times any repaired candidate; it is terminated after step 3,
  # on any error and on interruption.
  ensure_firesim "$ITERS iterations and the final verification"
  stage_problem
  (cd "$AC" && env "${autocomp_env[@]}" python -m autocomp.search.run_search) > "$OUT/2-search.log" 2>&1 \
    || die "Autocomp failed, see 2-search.log"

  sd=$(grep -o "Output directory: .*" "$OUT/2-search.log" | tail -1 | sed 's/Output directory: //')
  [ -d "$sd" ] || die "no Autocomp output directory in 2-search.log"
  echo "$sd" > "$OUT/search_dir"
  log "    search output: $sd"
  # The search starts from this run's Phase-1 kernel (code_0 of iteration 0). Autocomp's harness for the kernel
  # must accept it, e.g. read the weights in the harness's layout; if not, no candidate can be compared to it.
  python - "$sd" <<'EOF' || log "    WARNING: Autocomp's harness rejected the Phase-1 kernel (eval-results-iter-0/code_0_result.txt is not correct); see 2-search.log"
import glob, json, sys
f = glob.glob(f"{sys.argv[1]}/eval-results-iter-0/code_0_result.txt")
sys.exit(0 if f and json.load(open(f[0])).get("correct") else 1)
EOF
fi
[ "$STOP" -le 2 ] && { log "stopping after step 2 (STOP=2); continue with: E2E_OUT=$OUT START=3 $0 $K --vlen $VLEN"; exit 0; }

# ---------------------------------------------------------------- 3. verification
if [ "$START" -le 3 ]; then
  log "3/4 SMT: Neon vs Phase-1 RVV; the search's finalists, fastest first, vs Phase-1 RVV until one verifies, repairing faster ones that fail (${VS}s per proof)"
  sd=$(cat "$OUT/search_dir")
  rm -rf "$OUT/phase2" "$OUT/verify"
  # One proof run, as ae/verify.py runs it: elementwise kernels through artifact/elementwise.py with the manifest's
  # tagged specs (one after the other), multidimensional ones through growth_sweep.
  verify() {   # verify <phase> <RVV dir> <specs> <output dir>
    local p=$1 dir=$2 o=$4
    rm -rf "$o"; mkdir -p "$o"
    if [ "$FAMILY" = elementwise ]; then
      local envs=(SALT_AE_OUT="$o"); [ "$p" = 2 ] && envs+=(SALT_PHASE1_DIR="$OUT/phase1")
      (cd "$REPO/artifact" && env "${envs[@]}" python elementwise.py --phase "$p" --vlen "$VLEN" --seconds "$VS" \
         --workers 1 --rvv-dir "$dir" ${3//,/ })
    else
      local ref=(); [ "$p" = 2 ] && ref=(--reference-dir "$OUT/phase1")
      (cd "$REPO" && python -m src.workflow.verification.engine.growth_sweep --vlen "$VLEN" --seconds "$VS" --coverage \
         --rvv-dir "$dir" --kernel "$K" --output "$o/sweep" "${ref[@]}")
    fi
  }
  verify 1 "$OUT/phase1" "$SPECS1" "$OUT/verify/phase1/$K" > "$OUT/3-verify-phase1.log" 2>&1 &
  p1=$!
  # Phase 2's final verification (ae/phase2_final.py): finalists fastest first; a failed finalist faster than the
  # result is repaired with its counterexample (LLM), then compiled, checked on Spike against Phase 1, timed on
  # FireSim and proved. Exit 75: a repaired candidate needs FireSim timing and no run farm is up.
  stage_problem
  final=(python "$REPO/ae/phase2_final.py" --out "$OUT" --kernel "$K" --vlen "$VLEN" --family "$FAMILY" --specs2 "$SPECS2"
         --search-dir "$sd" --prob-type "$PROB" --prob-id "$PID" --seconds "$VS" --max-candidates "${MAX_CANDIDATES:-8}"
         --max-repairs "${MAX_REPAIRS:-2}" --repair-budget "${REPAIR_BUDGET:-4}"
         --model "${REPAIR_MODEL:-${PHASE1_MODEL:-gemini-3.8-flash}}")
  set +e
  env "${autocomp_env[@]}" "${final[@]}" $([ "$LAUNCHED" = 1 ] || echo --no-firesim) 2>&1 | tee -a "$OUT/e2e.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 75 ]; then
    set -e
    ensure_firesim "timing the repaired candidates"
    set +e
    env "${autocomp_env[@]}" "${final[@]}" 2>&1 | tee -a "$OUT/e2e.log"
    rc=${PIPESTATUS[0]}
  fi
  wait "$p1"
  set -e
  [ "$rc" = 0 ] || die "Phase 2's final verification failed (exit $rc), see 3-verify-phase2-*.log and 3-repair-autocomp.log"
  teardown
  restore_hw
fi
[ "$STOP" -le 3 ] && { log "stopping after step 3 (STOP=3); continue with: E2E_OUT=$OUT START=4 $0 $K --vlen $VLEN"; exit 0; }

# ---------------------------------------------------------------- 4. report
log "4/4 report"
python - "$OUT" "$K" "$REPO/artifact/expected" "$VLEN" "$FAMILY" <<'EOF' | tee -a "$OUT/e2e.log"
import csv, glob, json, os, subprocess, sys, tempfile
out, k, exp, vlen, family = sys.argv[1:]
summary = {"kernel": k, "vlen": int(vlen), "family": family}

def row(f):
    p = os.path.join(exp, f)
    return next((r for r in csv.DictReader(open(p)) if r["kernel"] == k), {}) if os.path.exists(p) else {}

st = json.load(open(f"{out}/kernels/status.json")).get(k, {}) if os.path.exists(f"{out}/kernels/status.json") else {}
summary["phase1_pipeline"] = {x: st.get(x) for x in ("compiled", "verified", "max_verified_batch", "attempts")}

# Step 3's proofs: the bound in the format of artifact/expected/final_vlen<V>_bounds.csv (artifact/bounds_table.py),
# and the outcome (VERIFIED, TIMEOUT or MAX_LEVEL_REACHED: no difference within the bound; else what stopped it).
d = {p: f"{out}/verify/phase{p}/{k}" for p in ("1", "2")}
bounds = {}
with tempfile.NamedTemporaryFile(suffix=".csv") as t:
    args = [sys.executable, "bounds_table.py", "--vlen", vlen, "--csv", t.name]
    for p in ("1", "2"):
        if os.path.isdir(d[p]):
            args += [f"--phase{p}", d[p]]
    if len(args) > 6 and subprocess.run(args, cwd=os.path.dirname(exp), stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL).returncode == 0:
        bounds = next((r for r in csv.DictReader(open(t.name)) if r["kernel"] == k), {})
bounds = {c: (b if b not in ("", "-") else None) for c, b in bounds.items()}
ok = ("VERIFIED", "TIMEOUT", "MAX_LEVEL_REACHED")
def outcome(dd):   # the worst outcome over a proof run's specs
    statuses = []
    for f in glob.glob(f"{dd}/elementwise_phase*/summary.json"):               # one entry per tagged spec
        for r in json.load(open(f)).values():
            statuses.append("ERROR" if r.get("stage") or r.get("error") else
                            (r.get("stopped") or [{}])[0].get("status") or "VERIFIED")
    for f in glob.glob(f"{dd}/sweep/summary.json"):
        statuses += [r.get("status", "ERROR") for r in json.load(open(f))]
    return next((s for s in statuses if s not in ok), None) or (max(statuses, key=ok.index) if statuses else "NOT RUN")
chosen = open(f"{out}/phase2/chosen").read().strip() if os.path.exists(f"{out}/phase2/chosen") else None
for p in ("1", "2"):
    status = outcome(d[p])
    summary[f"verify_phase{p}"] = {"reference": "Neon" if p == "1" else "Phase-1 RVV",
                                   "bound": bounds.get(f"phase{p}_bound") or None, "status": status,
                                   "verified": status in ok and bool(bounds.get(f"phase{p}_bound"))}
if chosen == "phase1":
    summary["verify_phase2"] = {"reference": "Phase-1 RVV", "bound": None, "status": "NO_VERIFIED_CANDIDATE",
                                "verified": False, "note": "no candidate verified; the Phase-1 kernel stays the result"}
# Which kernel became the Phase-2 kernel (ae/phase2_final.py): the finalists proved fastest first, and the repairs.
fin = json.load(open(f"{out}/phase2/final.json")) if os.path.exists(f"{out}/phase2/final.json") else {}
ch = fin.get("chosen") or {}
summary["phase2_selection"] = {
    "chosen": ch.get("name"), "kind": ch.get("kind"), "score": ch.get("score"),
    "finalists": [{"name": c["name"], "score": c["score"], "iter": c["iter"], "proof": fin.get("proofs", {}).get(c["name"], "not proved")}
                  for c in fin.get("candidates", [])],
    "repairs": [{x: r.get(x) for x in ("name", "of", "stage", "score", "proof", "detail", "error") if r.get(x) is not None}
                for r in fin.get("repairs", [])]}
sel = {"score": ch.get("score")} if ch.get("kind") in ("candidate", "repaired") else None

sd = open(f"{out}/search_dir").read().strip() if os.path.exists(f"{out}/search_dir") else ""
results = []
for f in glob.glob(f"{sd}/eval-results-iter-*/code_*_result.txt"):
    try:
        r = json.load(open(f))
    except (OSError, ValueError):
        continue
    if r.get("correct"):
        results.append((int(f.split("eval-results-iter-")[1].split("/")[0]), f, r))
base = next((r for it, f, r in results if it == 0 and f.endswith("code_0_result.txt")), None)
metric = "score" if base and base.get("score") is not None else "latency"
best = min((r for _, _, r in results if r.get(metric) is not None), key=lambda r: r[metric], default=None)
summary["search"] = {"dir": sd, "metric": metric, "correct_candidates": len(results),
                     "baseline": {x: base.get(x) for x in ("score", "latency", "firesim_latency")} if base else None,
                     "best": {x: best.get(x) for x in ("score", "latency", "firesim_latency")} if best else None}
if base and best and best.get(metric):
    summary["search"]["best_candidate_speedup"] = round(base[metric] / best[metric], 3)   # may be unverified
# The speedup of the Phase-2 kernel: the first candidate that verified (1.0 when none did, the Phase-1 kernel)
if base and base.get("score") and chosen:
    summary["search"]["speedup"] = round(base["score"] / sel["score"], 3) if sel else 1.0

# The provided kernels' results at this VLEN (artifact/expected/final_vlen<V>_*.csv), if they are there.
# Speedups there are over XNNPACK's RVV kernel; the search speedup above is over this run's Phase-1 kernel.
b, sp = row(f"final_vlen{vlen}_bounds.csv"), row(f"final_vlen{vlen}_speedups.csv")
if b or sp:
    summary["provided_kernels"] = {"phase1_bound": b.get("phase1_bound"), "phase2_bound": b.get("phase2_bound"),
                                   "phase1_speedup_over_xnnpack": sp.get("phase1"),
                                   "phase2_speedup_over_xnnpack": sp.get("phase2")}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
EOF
log "done: $OUT/summary.json"
