# ============================================================
# ベースイメージとシェル設定
# ============================================================
FROM python:3.10-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ============================================================
# ビルド引数（UID/GID/USERNAME）
# ============================================================
ARG UID=1000
ARG GID=1000
ARG USERNAME=devuser

# ============================================================
# OS基本ツールとロケール設定
# ============================================================
# 開発に必要な基本的なOSツールとSSH、日本語ロケールを設定
# dotfilesで入れるツール（vim, htop, tree, tmux等）は含めない
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash zsh git sudo build-essential wget curl locales gnupg \
    openssh-client openssh-server ca-certificates \
 && sed -i 's/^# *ja_JP.UTF-8/ja_JP.UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8 LANGUAGE=ja_JP:ja TZ=Asia/Tokyo TERM=xterm-256color

# ============================================================
# 非rootユーザー作成
# ============================================================
RUN groupadd --gid $GID $USERNAME \
 && useradd --uid $UID --gid $GID -m $USERNAME \
 && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME \
 && usermod --shell /usr/bin/zsh $USERNAME

# Git safe.directory設定
RUN git config --global --add safe.directory '*'

# SSH known_hosts事前設定
RUN mkdir -p /home/$USERNAME/.ssh \
 && ssh-keyscan github.com gitlab.com >> /home/$USERNAME/.ssh/known_hosts \
 && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

WORKDIR /home/$USERNAME/fluid-sbi


# ============================================================
# Python仮想環境の設定
# ============================================================
# venvを作成してPATH先頭に配置（activate不要にする）
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# 仮想環境の所有者をdevuserに変更
RUN chown -R $USERNAME:$USERNAME /opt/venv


# ============================================================
# Pythonパッケージマネージャー (uv)
# ============================================================
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version


# ============================================================
# 1Password CLI (dotfilesで使用)
# ============================================================
RUN curl -sS https://downloads.1password.com/linux/keys/1password.asc | \
    gpg --dearmor --output /usr/share/keyrings/1password-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/1password-archive-keyring.gpg] https://downloads.1password.com/linux/debian/amd64 stable main" | \
    tee /etc/apt/sources.list.d/1password.list && \
    apt-get update && apt-get install -y 1password-cli && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# Slurmワークロードマネージャーとその依存関係
# ============================================================
# クラスタ管理用のSlurmとMUNGE認証システム
RUN apt-get update && apt-get install -y --no-install-recommends \
    slurmctld slurmd slurm-client \
    munge libmunge2 pciutils && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# IBPM（埋め込み境界投影法）のビルド依存関係
# ============================================================
# FFTW3: 高速フーリエ変換ライブラリ（必須）
# doxygen: ドキュメント生成ツール（オプション）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-dev doxygen pkg-config && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# IBPMソースコードのクローンとビルド
# ============================================================
# IBPMを固定ディレクトリにクローンしてビルド
ENV IBPM_HOME=/opt/ibpm
RUN git clone --depth=1 https://github.com/cwrowley/ibpm.git $IBPM_HOME \
 && make -C $IBPM_HOME


# ============================================================
# IBPM実行用ラッパースクリプトの作成
# ============================================================
# どこからでも`ibpm`を実行可能にする
# $HOME/fluid-sbi/ibpmがある場合はそちらを優先ビルド→実行（拡張しやすい）
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'LOCAL_IBPM="${HOME}/fluid-sbi/ibpm"' \
    'if [[ -d "$LOCAL_IBPM" && -f "$LOCAL_IBPM/Makefile" ]]; then' \
    '  echo "[ibpm] $LOCAL_IBPM を検出。ローカルソースからビルドして実行します" >&2' \
    '  make -C "$LOCAL_IBPM" >/dev/null' \
    '  exec "$LOCAL_IBPM/build/ibpm" "$@"' \
    'else' \
    '  exec /opt/ibpm/build/ibpm "$@"' \
    'fi' \
    > /usr/local/bin/ibpm \
 && chmod +x /usr/local/bin/ibpm


# ============================================================
# Slurmエントリポイントスクリプトの配置
# ============================================================
# ctrl（コントローラ）、node（計算ノード）、dev（開発環境）用のエントリポイント
COPY infra/slurm/bin/entrypoint-ctrl.sh /usr/local/bin/entrypoint-ctrl.sh
COPY infra/slurm/bin/entrypoint-node.sh /usr/local/bin/entrypoint-node.sh
COPY infra/slurm/bin/entrypoint-dev.sh  /usr/local/bin/entrypoint-dev.sh
RUN chmod +x /usr/local/bin/entrypoint-*.sh


# ============================================================
# ポート公開
# ============================================================
# 8888: Jupyter, 6006: TensorBoard, 8000: 汎用Webサーバー
EXPOSE 8888 6006 8000

# ============================================================
# デフォルトユーザー
# ============================================================
USER $USERNAME
