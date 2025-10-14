FROM python:3.10-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# OS tools & locale
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash zsh git tmux build-essential wget curl vim htop tree locales \
    openssh-client openssh-server ca-certificates \
 && sed -i 's/^# *ja_JP.UTF-8/ja_JP.UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8 LANGUAGE=ja_JP:ja TZ=Asia/Tokyo TERM=xterm

WORKDIR /workspace

# venv を先に作って PATH 先頭へ（activate不要）
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH=/workspace:/workspace/sda

# uv
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

# Slurm/MUNGE
RUN apt-get update && apt-get install -y --no-install-recommends \
    slurmctld slurmd slurm-client \
    munge libmunge2 pciutils && \
    rm -rf /var/lib/apt/lists/*

# 必須: FFTW3（CPU版）。doxygen は任意（不要なら削ってOK）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-dev doxygen pkg-config && \
    rm -rf /var/lib/apt/lists/*

# ソースを固定ディレクトリへ配置してビルド
ENV IBPM_HOME=/opt/ibpm
RUN git clone --depth=1 https://github.com/cwrowley/ibpm.git $IBPM_HOME \
 && make -C $IBPM_HOME

# どこからでも `ibpm` 実行できるようにラッパーを用意
# /workspace/ibpm がある場合はそちらを優先ビルド→実行（拡張しやすい）
RUN bash -lc 'cat > /usr/local/bin/ibpm << "EOF"\n\
#!/usr/bin/env bash\n\
set -euo pipefail\n\
if [[ -d /workspace/ibpm && -f /workspace/ibpm/Makefile ]]; then\n\
  echo "[ibpm] /workspace/ibpm を検出。ローカルソースからビルドして実行します" >&2\n\
  make -C /workspace/ibpm >/dev/null\n\
  exec /workspace/ibpm/build/ibpm "$@"\n\
else\n\
  exec '"$IBPM_HOME"'/build/ibpm "$@"\n\
fi\n\
EOF\n\
chmod +x /usr/local/bin/ibpm'

# エントリポイントを同梱
COPY infra/slurm/bin/entrypoint-ctrl.sh /usr/local/bin/entrypoint-ctrl.sh
COPY infra/slurm/bin/entrypoint-node.sh /usr/local/bin/entrypoint-node.sh
COPY infra/slurm/bin/entrypoint-dev.sh  /usr/local/bin/entrypoint-dev.sh
RUN chmod +x /usr/local/bin/entrypoint-*.sh

# CLI快適化（任意）
RUN git clone https://github.com/ohmyzsh/ohmyzsh ~/.oh-my-zsh \
 && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
 && usermod --shell /usr/bin/zsh root \
 && sed -i 's/ZSH_THEME=".*"/ZSH_THEME="robbyrussell"/' ~/.zshrc \
 && sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc \
 && echo 'alias ll="ls -lah"' >> ~/.zshrc \
 && echo 'export EDITOR=vim' >> ~/.zshrc \
 && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
 && git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# lazygit (任意)
RUN LAZYGIT_VERSION=0.41.0 \
 && ARCH=$(dpkg --print-architecture | sed 's/amd64/x86_64/;s/arm64/aarch64/') \
 && curl -Lo /tmp/lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_${ARCH}.tar.gz" \
 && tar -C /tmp -xf /tmp/lazygit.tar.gz lazygit \
 && install /tmp/lazygit /usr/local/bin \
 && rm -f /tmp/lazygit.tar.gz /tmp/lazygit

# Node & claude-code（必要なら）
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm i -g @anthropic-ai/claude-code \
 && rm -rf /var/lib/apt/lists/*

EXPOSE 8888 6006 8000
