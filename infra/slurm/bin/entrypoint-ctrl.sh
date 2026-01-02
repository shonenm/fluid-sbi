#!/usr/bin/env bash
set -euo pipefail

# devuserで実行されるため、sudoを使用
sudo mkdir -p /etc/munge /var/log/munge /var/run/munge /var/spool/slurmctld /var/log/slurm

if [ ! -f /etc/munge/munge.key ]; then
  echo "[ctrl] generating /etc/munge/munge.key"
  sudo openssl rand -out /etc/munge/munge.key -base64 32
  sudo chmod 600 /etc/munge/munge.key
fi

sudo chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
# ディレクトリを755にして他コンテナがファイル存在確認できるようにする（キーは600のまま）
sudo chmod 755 /etc/munge
sudo chown -R "$(id -u):$(id -g)" /var/spool/slurmctld /var/log/slurm

sudo munged -f -v &
sleep 1

# SlurmUser=devuserに合わせてsudoなしで実行
exec slurmctld -Dvvv
