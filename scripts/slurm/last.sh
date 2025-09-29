#!/usr/bin/env bash
set -euo pipefail

JOBID=$(sacct --format=JobID --noheader -u "$USER" | awk 'NF' | tail -n 1)
if [[ -z "${JOBID:-}" ]]; then
  echo "No recent jobs."
  exit 0
fi

echo "JOBID: $JOBID"
scontrol show job "$JOBID"
