#!/usr/bin/env bash
# Multidimensional bounded verification, fully symbolic: full input domain, symbolic parameters within
# XNNPACK's constraints, VLEN 512 (or $VLEN), 600 s per kernel, XNNPACK's NaN rule, strict memory check,
# non-vacuity at level 1, threshold probes and coverage. Each kernel grows its shapes level by level.
# Every kernel is one single-threaded solver process, so at most one kernel runs per physical core
# (detected; SALT_JOBS overrides); the rest wait in one queue shared by all the sets being run.
#
#   multidim.sh phase1    Neon vs Phase-1 RVV ($SALT_PHASE1_DIR, default kernels/phase1/vlen<V>): 12 kernels
#   multidim.sh phase2    Phase-1 RVV ($SALT_PHASE1_DIR, default the reference kernels/phase2/vlen<V>/manifest.json
#                         names) vs Phase 2 ($SALT_PHASE2_DIR, default kernels/phase2/vlen<V>): 12 kernels
#   multidim.sh all       both
#
# Environment: SALT_AE_OUT (output root, default artifact/out), PYTHON (default python3), SECONDS_PER_KERNEL
# (default 600), SALT_JOBS (default: physical cores), VLEN (default 512, the paper's; any power of two >= 128).
# Output per set: $SALT_AE_OUT/multidim/<set>/summary.json (multidim-vlen<V>/ for a VLEN other than 512),
# plus parts/ (one run per kernel) and logs/. ./ae.sh verify runs the same sweeps for every provided set.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
VLEN="${VLEN:-512}"
OUT="${SALT_AE_OUT:-$HERE/out}/multidim"
[ "$VLEN" = 512 ] || OUT="$OUT-vlen$VLEN"
PY="${PYTHON:-python3}"
BUDGET="${SECONDS_PER_KERNEL:-600}"
ALL=(f32-gemm-minmax f32-igemm-minmax qd8-f32-qc4w-gemm-minmax f32-dwconv-minmax f32-conv-hwc2chw f32-argmaxpool
     f32-maxpool f32-dwconv2d-chw qs8-rdsum qu8-rdsum x32-packw-gemm-goi x32-transposec)
# A kernel waits for a free core before it compiles and starts its timer.
JOBS="${SALT_JOBS:-$(cd "$HERE" && "$PY" paths.py jobs)}"
REF="${SALT_PHASE1_DIR:-$REPO/$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["reference"])' \
                                  "$REPO/kernels/phase2/vlen$VLEN/manifest.json")}"

declare -A SET_DIR SET_KERNELS SET_REF
SET_DIR[phase1]="${SALT_PHASE1_DIR:-$REPO/kernels/phase1/vlen$VLEN}"; SET_KERNELS[phase1]="${ALL[*]}"
SET_DIR[phase2]="${SALT_PHASE2_DIR:-$REPO/kernels/phase2/vlen$VLEN}"; SET_KERNELS[phase2]="${ALL[*]}"
SET_REF[phase2]="$REF"                                    # reference: Phase-1 RVV instead of Neon

run_one() {   # run_one <set> <rvv dir or ""> <kernel>: one growth_sweep in its own directory
  local name=$1 dir=$2 k=$3 rc=0
  local extra=()
  [ -n "$dir" ] && extra=(--rvv-dir "$dir")
  [ -n "${SET_REF[$name]:-}" ] && extra+=(--reference-dir "${SET_REF[$name]}")
  # growth_sweep builds in build/verification_cvc5/<output name>-<hash of the output path>: one per run.
  (cd "$REPO" && "$PY" -m src.workflow.verification.engine.growth_sweep --vlen "$VLEN" --seconds "$BUDGET" --coverage \
     ${extra[@]+"${extra[@]}"} --kernel "$k" --output "$OUT/$name/parts/$name.$k") > "$OUT/$name/logs/$k.log" 2>&1 \
    || rc=$?
  if [ "$rc" -eq 0 ]; then echo "[$name] $k done"; else echo "[$name] $k exited with $rc (see $OUT/$name/logs/$k.log)"; fi
}

merge() {     # merge <set> <kernel...>: the set's summary.json in kernel order, as one growth_sweep run wrote it
  local name=$1; shift
  "$PY" - "$OUT/$name" "$@" <<'EOF'
import json, sys
from pathlib import Path
out, kernels = Path(sys.argv[1]), sys.argv[2:]
rows = []
for k in kernels:
    s = out / "parts" / f"{out.name}.{k}" / "summary.json"
    rows += json.loads(s.read_text()) if s.exists() else [dict(kernel=k, status="NO_RESULT", verified_shapes=0,
                                                              completed_level=0, methods={})]
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"[{out.name}] summary: {out / 'summary.json'}")
EOF
}

case "${1:-}" in
  phase1|phase2) SETS=("$1") ;;
  all) SETS=(phase1 phase2) ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac

total=0
for s in "${SETS[@]}"; do
  rm -rf "${OUT:?}/$s"
  mkdir -p "$OUT/$s/parts" "$OUT/$s/logs"
  read -ra ks <<< "${SET_KERNELS[$s]}"
  total=$((total + ${#ks[@]}))
done
echo "${SETS[*]}: $total kernels at VLEN $VLEN, ${BUDGET}s each, $JOBS at once (one per physical core) -> $OUT"
for s in "${SETS[@]}"; do
  read -ra ks <<< "${SET_KERNELS[$s]}"
  for k in "${ks[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
    run_one "$s" "${SET_DIR[$s]}" "$k" &
  done
done
wait || true
for s in "${SETS[@]}"; do
  read -ra ks <<< "${SET_KERNELS[$s]}"
  merge "$s" "${ks[@]}"
done
