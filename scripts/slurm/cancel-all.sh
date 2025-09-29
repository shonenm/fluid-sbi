#!/usr/bin/env bash
set -euo pipefail
JOBS=$(squeue --me --noheader -o "%i")
[[ -z "$JOBS" ]] && { echo "No jobs."; exit 0; }
echo "$JOBS" | xargs -r scancel
