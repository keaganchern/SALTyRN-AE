#!/usr/bin/env bash
# fs_wait_ssh.sh <FireSim deploy dir>
#
# Wait until every running instance of the run farm (tag fsimcluster = config_runtime.yaml's run_farm_tag) accepts
# a key-based SSH session three times in a row. `firesim launchrunfarm` returns once the instance is "running", but
# sshd listens before the instance has finished setting up its login: for about the first 15-20 s a session is
# dropped or asks for a password, and `firesim infrasetup` (Fabric in parallel mode) aborts on that instead of
# retrying ("Connect error: No existing session", "Needed to prompt for a connection or sudo password").
# Used by ae/e2e.sh and ae/speedups/run_firesim.sh right after launchrunfarm.
#
# Environment: FIRESIM_KEY (default ~/firesim.pem), FS_SSH_WAIT (seconds, default 600).
set -u
d=${1:?usage: fs_wait_ssh.sh <FireSim deploy dir>}
key=${FIRESIM_KEY:-$HOME/firesim.pem}
tag=$(sed -n 's/^\s*run_farm_tag:\s*\(\S*\).*/\1/p' "$d/config_runtime.yaml" | head -1)
[ -n "$tag" ] || { echo "fs_wait_ssh: no run_farm_tag in $d/config_runtime.yaml" >&2; exit 1; }
region=$(aws configure get region 2> /dev/null); region=${region:-us-east-1}
deadline=$((SECONDS + ${FS_SSH_WAIT:-600}))
ips=""
while [ -z "$ips" ]; do
  ips=$(aws ec2 describe-instances --region "$region" --output text \
          --filters "Name=tag:fsimcluster,Values=$tag" "Name=instance-state-name,Values=running" \
          --query 'Reservations[].Instances[].PrivateIpAddress' 2> /dev/null)
  [ -n "$ips" ] && break
  [ $SECONDS -lt $deadline ] || { echo "fs_wait_ssh: no running instance tagged fsimcluster=$tag" >&2; exit 1; }
  sleep 5
done
for ip in $ips; do
  ok=0 tries=0
  while [ $ok -lt 3 ]; do
    [ $SECONDS -lt $deadline ] || { echo "fs_wait_ssh: $ip: no stable SSH session within ${FS_SSH_WAIT:-600} s" >&2; exit 1; }
    tries=$((tries + 1))
    if ssh -i "$key" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
           -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$ip" true 2> /dev/null; then
      ok=$((ok + 1)); sleep 3
    else
      ok=0; sleep 5
    fi
  done
  echo "fs_wait_ssh: $ip accepts SSH (3 sessions in a row, $tries attempts)"
done
