#!/usr/bin/env bash
set -euo pipefail

# ワークスペースの.envを読み込む
WORKSPACE_ENV="$HOME/fluid-sbi/.env"
if [ -f "$WORKSPACE_ENV" ]; then
  echo "[dev] loading $WORKSPACE_ENV"
  set -a
  source "$WORKSPACE_ENV"
  set +a
fi

# SSHエージェントソケット（コンテナ内の固定パス）
export SSH_AUTH_SOCK=/ssh-agent/agent.sock
echo "[dev] SSH_AUTH_SOCK: $SSH_AUTH_SOCK"

# devuserで実行されるため、sudoを使用
sudo mkdir -p /etc/munge /var/log/munge /var/run/munge

# mungeキーが存在するまで待機（最大30秒）
# ctrlコンテナがmungeキーを生成する責任を持つ
echo "[dev] waiting for munge.key..."
for i in {1..30}; do
  if [ -f /etc/munge/munge.key ]; then
    echo "[dev] munge.key found"
    break
  fi
  sleep 1
done

if [ ! -f /etc/munge/munge.key ]; then
  echo "[dev] WARNING: munge.key not found, generating locally (SLURM may not work correctly)"
  sudo openssl rand -out /etc/munge/munge.key -base64 32
  sudo chmod 600 /etc/munge/munge.key
fi

sudo chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
sudo munged -f -v &

echo "[dev] ready."
exec zsh -l
