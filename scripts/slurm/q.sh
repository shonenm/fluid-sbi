#!/usr/bin/env bash
set -euo pipefail
squeue --me -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
