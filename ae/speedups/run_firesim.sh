#!/usr/bin/env bash
# Time the perf_compare ELFs from `./ae.sh speedups prepare` on FireSim (Saturn + Shuttle on one AWS f2.6xlarge).
#
#   ./ae.sh speedups run [--vlen 128,512] [group|vlen:group ...]   default: both VLENs, every group prepare built
#
# One run farm for the whole run. For each ELF: the FireSim image of its VLEN (GENV128D64 or GENV512D256), infrasetup,
# runworkload, the uartlog polled over SSH; done at "BENCHMARK COMPLETE", stopped on a trap (mcause), when the uartlog
# has not grown for STALL seconds or after CAP seconds. The uartlogs go to $SALT_AE_OUT/speedups/vlen<V>/firesim/.
# config_runtime.yaml's default_hw_config and workload, and the workload's boot binary, are changed for the run and
# restored at the end. The run farm is terminated at the end, on any error and on interruption.
#
# Environment
#   CHIPYARD_PATH  Chipyard with FireSim set up for F2 and the two Saturn images in config_hwdb.yaml (./ae.sh setup);
#                  default ~/chipyard
#   FIRESIM_KEY    the run farm's SSH key (default ~/firesim.pem)
#   YES=1          do not ask before launching the f2 instance
#   CAP, STALL     per-ELF limits in seconds (default 7200, 1200)
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "${SALT_AE_OUT:-$REPO/out}")/speedups
CY=${CHIPYARD_PATH:-$HOME/chipyard}
FS=$CY/sims/firesim
WL=$FS/deploy/workloads/saturn/saturn_test-baremetal
RT=$FS/deploy/config_runtime.yaml
KEY=${FIRESIM_KEY:-$HOME/firesim.pem}
CAP=${CAP:-7200}
STALL=${STALL:-1200}
declare -A HW=([128]=firesim_saturn_GENV128D64_shuttle_singlecore_no_nic_l2_llc4mb_ddr3
               [512]=firesim_saturn_GENV512D256_shuttle_singlecore_no_nic_l2_llc4mb_ddr3)
VLENS=128,512
SEL=()
while [ $# -gt 0 ]; do
  case $1 in
    --vlen) VLENS=$2; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) SEL+=("$1"); shift ;;
  esac
done
mkdir -p "$OUT/firesim"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OUT/firesim/run.log"; }
die() { log "ERROR: $*"; exit 1; }
# Run a command in the FireSim manager environment. sourceme-manager.sh is not written for `set -u` and must not see
# this function's arguments; FireSim runs on Chipyard's conda Python, not SALTyRN's venv.
fs() { local cmd=("$@")
       ( set +eu; set --; export PATH="${PATH//$REPO\/.venv\/bin:/}"
         cd "$FS" && source sourceme-manager.sh > /dev/null 2>&1 && command -v firesim > /dev/null && "${cmd[@]}" ); }
ssh_run() { ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
                -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$HOST" "$@"; }

# The queue: (VLEN, group) for every ELF prepare built.
QUEUE=()
for v in ${VLENS//,/ }; do
  [ -n "${HW[$v]:-}" ] || die "no FireSim image for VLEN $v"
  d=$OUT/vlen$v
  if [ ${#SEL[@]} = 0 ]; then
    for b in "$d"/build-*/zephyr/zephyr.elf; do [ -f "$b" ] && QUEUE+=("$v:$(basename "$(dirname "$(dirname "$b")")" | sed 's/^build-//')"); done
  else
    for g in "${SEL[@]}"; do              # a group name, or <vlen>:<group> for one VLEN only
      if [[ $g == *:* ]]; then [ "${g%%:*}" = "$v" ] && QUEUE+=("$g"); else QUEUE+=("$v:$g"); fi
    done
  fi
done
[ ${#QUEUE[@]} -gt 0 ] || die "no ELFs in $OUT/vlen*/build-*: run ./ae.sh speedups prepare first"
for q in "${QUEUE[@]}"; do
  [ -f "$OUT/vlen${q%%:*}/build-${q#*:}/zephyr/zephyr.elf" ] || die "no ELF for $q (run ./ae.sh speedups prepare)"
done
[ -f "$FS/sourceme-manager.sh" ] || die "no FireSim in $FS (set CHIPYARD_PATH)"
[ -f "$KEY" ] || die "no SSH key $KEY (set FIRESIM_KEY)"
for v in ${VLENS//,/ }; do
  grep -q "^${HW[$v]}:" "$FS/deploy/config_hwdb.yaml" || die "${HW[$v]} is not in config_hwdb.yaml (run ./ae.sh setup)"
done
[ -f "$FS/deploy/workloads/saturn.json" ] && [ -d "$(dirname "$WL")" ] || die "no saturn.json workload (run ./ae.sh setup)"

acct=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || die "no AWS credentials"
region=$(aws configure get region 2>/dev/null); region=${region:-us-east-1}
log "AWS account $acct ($region); ELFs: ${QUEUE[*]}"
log "launching 1 x f2.6xlarge, On-Demand (about \$2/hour; about 1 hour at VLEN 128 and 2.5 hours at VLEN 512)"
if [ "${YES:-0}" != 1 ]; then
  read -r -p "continue? [y/N] " a; [ "$a" = y ] || [ "$a" = Y ] || die "not confirmed"
fi

cp "$RT" "$OUT/firesim/config_runtime.yaml.before"
[ -f "$WL" ] && cp "$WL" "$OUT/firesim/saturn_test-baremetal.before"
restore() { cp "$OUT/firesim/config_runtime.yaml.before" "$RT"
            [ -f "$OUT/firesim/saturn_test-baremetal.before" ] && cp "$OUT/firesim/saturn_test-baremetal.before" "$WL"; }
teardown() {
  fs firesim kill >> "$OUT/firesim/firesim.log" 2>&1
  fs firesim terminaterunfarm --forceterminate >> "$OUT/firesim/firesim.log" 2>&1
  for _ in $(seq 12); do                 # terminated instances pass through shutting-down: wait for them
    left=$(aws ec2 describe-instances --region "$region" --filters "Name=instance-type,Values=f2.*" \
           "Name=instance-state-name,Values=pending,running" --query 'Reservations[].Instances[].InstanceId' --output text 2>&1)
    [ -z "$left" ] && break
    sleep 10
  done
  log "teardown: f2 instances still running: ${left:-none}"
  restore
}
trap 'log "interrupted"; teardown; exit 130' INT TERM HUP
sed -i "s/^\(\s*workload_name:\).*/\1 saturn.json/" "$RT"

fs firesim launchrunfarm >> "$OUT/firesim/firesim.log" 2>&1 || { log "ERROR: launchrunfarm failed"; teardown; exit 1; }
# sshd answers before the instance accepts logins; the first infrasetup would abort on that
"$REPO/ae/fs_wait_ssh.sh" "$FS/deploy" >> "$OUT/firesim/firesim.log" 2>&1 \
  || { log "ERROR: the f2 instance did not accept SSH sessions, see firesim.log"; teardown; exit 1; }
log "run farm launched, SSH ready"
for q in "${QUEUE[@]}"; do
  v=${q%%:*}; e=${q#*:}; d=$OUT/vlen$v; mkdir -p "$d/firesim"
  sed -i "s/^\(\s*default_hw_config:\).*/\1 ${HW[$v]}/" "$RT"
  cp "$d/build-$e/zephyr/zephyr.elf" "$WL"
  touch "$OUT/firesim/.infrasetup-start"
  o=$(fs firesim infrasetup 2>&1); rc=$?; echo "$o" >> "$OUT/firesim/firesim.log"
  [ $rc = 0 ] || { log "ERROR: $q: infrasetup exited $rc"; teardown; exit 1; }
  ilog=$(find "$FS/deploy/logs" -name '*-infrasetup-*.log' -newer "$OUT/firesim/.infrasetup-start" 2>/dev/null | head -1)
  HOST=$( { echo "$o"; [ -n "$ilog" ] && cat "$ilog"; } | grep -oE "\[[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\]" | tr -d '[]' \
          | grep -v '^127\.' | sort -u | head -1)
  [ -n "$HOST" ] || { log "ERROR: $q: no run farm host in the infrasetup output"; teardown; exit 1; }
  ssh_run ": > /home/ubuntu/sim_slot_0/uartlog" 2>/dev/null
  log "$q: infrasetup done (${HW[$v]}, host $HOST), runworkload"
  ( fs firesim runworkload >> "$OUT/firesim/runworkload-$v-$e.log" 2>&1 ) < /dev/null &
  rw=$!; t0=$(date +%s); last=0; grow=$t0; status=ended
  u=$d/firesim/$e.uartlog
  while kill -0 $rw 2> /dev/null; do
    sleep 30
    ssh_run "cat /home/ubuntu/sim_slot_0/uartlog" > "$u.part" 2>/dev/null     # a copy, in case the host is lost
    n=$(wc -c < "$u.part")
    [ "${n:-0}" -gt "$last" ] && { last=$n; grow=$(date +%s); }
    if grep -q 'BENCHMARK COMPLETE' "$u.part"; then status=complete; break; fi
    if grep -q 'mcause' "$u.part"; then status=crash; break; fi
    now=$(date +%s)
    [ $((now - grow)) -gt "$STALL" ] && { status=stall; break; }
    [ $((now - t0)) -gt "$CAP" ] && { status=cap; break; }
  done
  sleep 5
  ssh_run "cat /home/ubuntu/sim_slot_0/uartlog" > "$u" 2>/dev/null
  [ -s "$u" ] || cp "$u.part" "$u"
  rm -f "$u.part"
  kill -0 $rw 2> /dev/null && fs firesim kill >> "$OUT/firesim/firesim.log" 2>&1
  wait $rw 2> /dev/null
  log "$q: $status after $(( $(date +%s) - t0 ))s; $(grep -c '^CSV:' "$u") CSV rows -> $u"
done
teardown
log "all done; next: ./ae.sh speedups analyze"
