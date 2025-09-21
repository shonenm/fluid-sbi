# Fluid SBI (Score-based Data Assimilation) Development Environment
FROM python:3.10-slim

# 以降の RUN を bash で実行（pipefail で堅牢化）
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ===== OS 基本ツール & ロケール =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash zsh git tmux build-essential wget curl vim htop tree locales \
    openssh-client openssh-server ca-certificates \
 && sed -i 's/^# *ja_JP.UTF-8/ja_JP.UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8 \
    LC_ALL=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    TZ=Asia/Tokyo \
    TERM=xterm

WORKDIR /workspace

# ===== Python仮想環境（venv） =====
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/workspace:/workspace/sda

# uv（高速pip互換）※任意
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ===== Python依存（論文期互換） =====
# * CPU版 JAX: 0.4.30 系を固定
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
      "jax==0.4.30" "jaxlib==0.4.30" \
      numpy scipy pandas matplotlib seaborn jupyterlab ipywidgets tqdm wandb pot \
 && pip install git+https://github.com/google/jax-cfd

# ===== プロジェクト配置 & sda を開発インストール =====
COPY . .
RUN pip install -e ./sda

# ===== oh-my-zsh & プラグイン =====
RUN git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh \
 && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
 && usermod --shell /usr/bin/zsh root \
 && sed -i 's/ZSH_THEME=".*"/ZSH_THEME="robbyrussell"/' ~/.zshrc \
 && sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc \
 && echo 'alias ll="ls -lah"' >> ~/.zshrc \
 && echo 'export EDITOR=vim' >> ~/.zshrc \
 && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
 && git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
 # venv を zsh でも確実に使う（PATHはENVで通っているが念のため明示）
 && printf '\n# --- venv auto-use ---\nexport VIRTUAL_ENV=/opt/venv\nexport PATH="/opt/venv/bin:$PATH"\n' >> ~/.zshrc

# あなたの .tmux.conf / .zshrc を使いたい場合は次を有効化（存在しないとビルド失敗するので注意）
# COPY .tmux.conf /root/.tmux.conf
# COPY .zshrc /root/.zshrc

# ===== lazygit =====
RUN LAZYGIT_VERSION=0.41.0 \
 && ARCH=$(dpkg --print-architecture | sed 's/amd64/x86_64/;s/arm64/aarch64/') \
 && curl -Lo /tmp/lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_${ARCH}.tar.gz" \
 && tar -C /tmp -xf /tmp/lazygit.tar.gz lazygit \
 && install /tmp/lazygit /usr/local/bin \
 && rm -f /tmp/lazygit.tar.gz /tmp/lazygit

# ===== SSH（必要な人だけ）=====
RUN mkdir -p /var/run/sshd \
 && echo 'root:Docker!' | chpasswd \
 && sed -i 's/#PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's@#AuthorizedKeysFile.*@AuthorizedKeysFile /root/.ssh/authorized_keys@' /etc/ssh/sshd_config

# ===== Node.js & claude-code（任意）=====
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm i -g @anthropic-ai/claude-code \
 && rm -rf /var/lib/apt/lists/*

# ポート（Dev Containers のポート転送を使用）
EXPOSE 8888 6006 8000
