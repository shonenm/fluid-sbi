#!/usr/bin/env bash
set -euo pipefail
mkdir -p /etc/munge /var/log/munge /var/run/munge
if [ ! -f /etc/munge/munge.key ]; then
  echo "[dev] generating /etc/munge/munge.key"
  openssl rand -out /etc/munge/munge.key -base64 32
  chmod 600 /etc/munge/munge.key
fi
chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
munged -f -v &
echo "[dev] ready."
exec zsh -l
