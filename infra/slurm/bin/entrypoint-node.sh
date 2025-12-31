#!/usr/bin/env bash
set -euo pipefail

# devuserで実行されるため、sudoを使用
sudo mkdir -p /etc/munge /var/log/munge /var/run/munge /var/spool/slurmd /var/log/slurm

sudo chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
sudo chown -R "$(id -u):$(id -g)" /var/spool/slurmd /var/log/slurm

sudo munged -f -v &
sleep 1

exec sudo slurmd -Dvvv
