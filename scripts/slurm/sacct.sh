#!/usr/bin/env bash
set -euo pipefail

N="${1:-20}"
# PENDING/FAILED/SUCCESS の履歴を簡易表示
sacct -u "$USER" --format=JobID,JobName%25,State,Elapsed,ReqCPUS,ReqMem,Submit,Start,End --parsable2 --noheader \
| tail -n "$N" \
| column -t -s'|'
