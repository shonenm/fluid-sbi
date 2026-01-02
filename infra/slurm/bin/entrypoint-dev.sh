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

if [ ! -f /etc/munge/munge.key ]; then
  echo "[dev] generating /etc/munge/munge.key"
  sudo openssl rand -out /etc/munge/munge.key -base64 32
  sudo chmod 600 /etc/munge/munge.key
fi

sudo chown -R munge:munge /etc/munge /var/log/munge /var/run/munge || true
sudo munged -f -v &

echo "[dev] ready."
exec zsh -l
