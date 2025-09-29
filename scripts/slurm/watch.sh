#!/usr/bin/env bash
set -euo pipefail
watch -n 2 "squeue --me -o '%.18i %.9P %.20j %.2t %.10M %.6D %R'"
