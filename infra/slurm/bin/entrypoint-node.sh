#!/usr/bin/env bash
set -euo pipefail
mkdir -p /etc/munge /var/log/munge /var/run/munge /var/spool/slurmd /var/log/slurm
chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
chown -R root:root /var/spool/slurmd /var/log/slurm
munged -f -v &
sleep 1
exec slurmd -Dvvv
