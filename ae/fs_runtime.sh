#!/usr/bin/env bash
# fs_runtime.sh get <deploy dir>            print config_runtime.yaml's default_hw_config
# fs_runtime.sh set <deploy dir> <image>    set it (the image must be in config_hwdb.yaml)
# fs_runtime.sh check <deploy dir> <image>  exit 0 if the image is in config_hwdb.yaml and the workload is saturn.json
# Used by ae/e2e.sh to run on the FPGA image of the requested VLEN and put the previous one back afterwards.
set -euo pipefail
cmd=$1 d=$2
rt=$d/config_runtime.yaml
case "$cmd" in
  get) sed -n 's/^\s*default_hw_config:\s*\(\S*\).*/\1/p' "$rt" | head -1 ;;
  set) grep -q "^$3:" "$d/config_hwdb.yaml" || { echo "no $3 in $d/config_hwdb.yaml" >&2; exit 1; }
       sed -i "s/^\(\s*default_hw_config:\).*/\1 $3/" "$rt" ;;
  check) grep -q "^$3:" "$d/config_hwdb.yaml" && grep -q "^\s*workload_name:\s*saturn.json" "$rt" ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac
