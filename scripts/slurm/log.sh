#!/usr/bin/env bash
set -euo pipefail

# 検索ディレクトリ: 環境変数で上書き可
SEARCH_DIRS="${SLURM_LOG_SEARCH_DIRS:-/workspace/logs .}"

find_latest_log() {
  # 新しい順で slurm-*.out を探す
  for dir in $SEARCH_DIRS; do
    if [[ -d "$dir" ]]; then
      LATEST=$(ls -1t "$dir"/slurm-*.out 2>/dev/null | head -n 1 || true)
      [[ -n "${LATEST:-}" ]] && { echo "$LATEST"; return 0; }
    fi
  done
  return 1
}

find_by_jobid() {
  local jobid="$1"
  for dir in $SEARCH_DIRS; do
    if [[ -d "$dir" ]]; then
      CAND=$(ls -1t "$dir"/slurm-"$jobid"* 2>/dev/null | head -n 1 || true)
      [[ -n "${CAND:-}" ]] && { echo "$CAND"; return 0; }
    fi
  done
  return 1
}

if [[ $# -eq 0 ]]; then
  FILE=$(find_latest_log) || { echo "No slurm-*.out found in: $SEARCH_DIRS"; exit 1; }
  echo "tail -f $FILE"
  tail -f "$FILE"
else
  JOBID="$1"
  FILE=$(find_by_jobid "$JOBID") || { echo "No logs for JOBID=$JOBID in: $SEARCH_DIRS"; exit 1; }
  echo "tail -f $FILE"
  tail -f "$FILE"
fi
