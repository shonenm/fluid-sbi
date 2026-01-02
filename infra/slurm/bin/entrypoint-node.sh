#!/usr/bin/env bash
set -euo pipefail

# devuserで実行されるため、sudoを使用
sudo mkdir -p /etc/munge /var/log/munge /var/run/munge /var/spool/slurmd /var/log/slurm

# mungeキーが存在するまで待機（最大30秒）
# ctrlコンテナがmungeキーを生成する責任を持つ
echo "[node] waiting for munge.key..."
for i in {1..30}; do
  if [ -f /etc/munge/munge.key ]; then
    echo "[node] munge.key found"
    break
  fi
  sleep 1
done

if [ ! -f /etc/munge/munge.key ]; then
  echo "[node] ERROR: munge.key not found after 30s" >&2
  exit 1
fi

sudo chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
sudo chown -R "$(id -u):$(id -g)" /var/spool/slurmd /var/log/slurm

sudo munged -f -v &
sleep 1

# SlurmdUser=devuserに合わせてsudoなしで実行
exec slurmd -Dvvv
