#!/usr/bin/env bash
set -euo pipefail
mkdir -p /etc/munge /var/log/munge /var/run/munge /var/spool/slurmctld /var/log/slurm
if [ ! -f /etc/munge/munge.key ]; then
  echo "[ctrl] generating /etc/munge/munge.key"
  openssl rand -out /etc/munge/munge.key -base64 32
  chmod 600 /etc/munge/munge.key
fi
chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
chown -R root:root /var/spool/slurmctld /var/log/slurm
munged -f -v &
sleep 1
exec slurmctld -Dvvv
