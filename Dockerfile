# Fluid SBI (Score-based Data Assimilation) Development Environment
FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    zsh \
    git \
    tmux \
    build-essential \
    wget \
    curl \
    vim \
    htop \
    tree \
    locales \
    openssh-client \
    openssh-server \
 && localedef -i ja_JP -f UTF-8 ja_JP.UTF-8 \
 && update-locale LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

 # ロケール & タイムゾーン設定
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=Asia/Tokyo
ENV TERM=xterm

# Set working directory
WORKDIR /workspace

# Copy SDA environment file (from submodule)
COPY sda/environment.yml* ./

# mamba で高速＆衝突低減（任意だが強く推奨）
RUN conda install -n base -c conda-forge -y mamba \
 && conda config --set channel_priority strict

# ★ ここが肝：環境ファイルで env 作成 → jax-cfd は別途 pip
#   もし upstream の environment.yml が jax/jaxlib==0.4.4 を固定していて
#   ビルドが転ぶ場合に備え、0.4.30 へ置換するフォールバックを入れておく
RUN if [ -f environment.yml ]; then \
      sed -E '/^[[:space:]]*-[[:space:]]*jax==/d; /^[[:space:]]*-[[:space:]]*jaxlib==/d' environment.yml > environment.patched.yml; \
      mamba env create -n sda -f environment.patched.yml; \
      conda run -n sda python -m pip install --upgrade pip && \
      conda run -n sda pip install --no-cache-dir "jax==0.4.30" "jaxlib==0.4.30"; \
    else \
      mamba create -n sda -y python=3.9.16; \
      mamba install -n sda -c conda-forge -y \
        scipy numpy matplotlib pandas seaborn jupyterlab ipywidgets; \
      mamba install -n sda -c pytorch -c nvidia -y \
        pytorch torchvision torchaudio pytorch-cuda=11.8; \
      conda run -n sda python -m pip install --upgrade pip && \
      conda run -n sda pip install --no-cache-dir "jax==0.4.30" "jaxlib==0.4.30"; \
    fi

# 著者推奨：jax-cfd は Git から（READMEの指示通り）
RUN conda run -n sda pip install git+https://github.com/google/jax-cfd

# プロジェクトコピー & sda を開発インストール（著者推奨）
COPY . .
RUN conda run -n sda pip install -e ./sda

# ディレクトリ
RUN mkdir -p /workspace/data/{inputs,outputs,raw,processed} \
 && mkdir -p /workspace/results/{models,figures,logs}

# SSHサーバーの設定
RUN mkdir -p /var/run/sshd \
 && echo 'root:Docker!' | chpasswd \
 && sed -i 's/#PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's@#AuthorizedKeysFile.*@AuthorizedKeysFile /root/.ssh/authorized_keys@' /etc/ssh/sshd_config

# oh-my-zsh インストールと設定
RUN git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh && \
    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc && \
    usermod --shell /usr/bin/zsh root && \
    sed -i 's/ZSH_THEME=".*"/ZSH_THEME="robbyrussell"/' ~/.zshrc && \
    echo 'alias ll="ls -lah"' >> ~/.zshrc && \
    echo 'export EDITOR=vim' >> ~/.zshrc && \
    echo 'setopt AUTO_CD' >> ~/.zshrc && \
    echo 'setopt HIST_IGNORE_ALL_DUPS' >> ~/.zshrc

# プラグイン：補完 & syntax highlight
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# lazygit インストール（固定バージョン & 自動アーキ対応）
RUN LAZYGIT_VERSION=0.41.0 && \
    ARCH=$(dpkg --print-architecture | sed 's/amd64/x86_64/;s/arm64/aarch64/') && \
    curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_${ARCH}.tar.gz" && \
    tar xf lazygit.tar.gz lazygit && \
    install lazygit /usr/local/bin && \
    rm -f lazygit.tar.gz lazygit

# ★ .tmux.conf をコンテナ内にコピー
COPY .tmux.conf /root/.tmux.conf

# .zshrc をプロジェクトからコピー（テンプレート編集は不要）
COPY .zshrc /root/.zshrc

# conda の hook を恒久化（bash / zsh 両対応）
RUN ln -sf /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/zsh/zshrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/profile

# Node.js（NodeSource経由）
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# claude 設定
RUN npm install -g @anthropic-ai/claude-code

# docker compose プラグインのインストール
RUN mkdir -p /usr/libexec/docker/cli-plugins && \
    curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 \
    -o /usr/libexec/docker/cli-plugins/docker-compose && \
    chmod +x /usr/libexec/docker/cli-plugins/docker-compose

# uvのインストール (for serena)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 環境変数設定
ENV PATH="/root/.local/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=sda
ENV PATH=/opt/conda/envs/sda/bin:/opt/conda/bin:/root/.local/bin:$PATH
ENV PYTHONPATH=/workspace:/workspace/sda

EXPOSE 8888 6006 8000
