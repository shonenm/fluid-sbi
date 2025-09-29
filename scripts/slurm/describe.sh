#!/usr/bin/env bash
set -euo pipefail

JOBID="${1:-}"
if [[ -z "$JOBID" ]]; then
  echo "Usage: $(basename "$0") <JOBID>"
  exit 1
fi

scontrol show job "$JOBID"
echo "---- sstat (if running) ----"
# 実行中ならリソース実績
sstat -j "${JOBID}.batch" --format=AveCPU,AveRSS,MaxRSS,Elapsed 2>/dev/null || true
