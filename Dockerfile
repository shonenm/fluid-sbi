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
    openssh-client \
    openssh-server \
    locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 && \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy SDA environment file (from submodule)
COPY sda/environment.yml* ./

# Create conda environment
RUN if [ -f environment.yml ]; then \
        conda env create -f environment.yml; \
    else \
        conda create -n sda python=3.9 -y && \
        conda install -n sda pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
        conda install -n sda jax jaxlib -c conda-forge -y && \
        conda install -n sda scipy numpy matplotlib pandas seaborn jupyterlab ipywidgets -c conda-forge -y && \
        conda run -n sda pip install pot wandb tqdm; \
    fi

# Install jax-cfd separately as required
RUN conda run -n sda pip install git+https://github.com/google/jax-cfd

# Copy all project files
COPY . .

# Install SDA package in development mode
RUN conda run -n sda pip install -e ./sda

# Set up bash to automatically activate conda environment
RUN echo "conda activate sda" >> ~/.bashrc
RUN echo "export PYTHONPATH=/workspace:/workspace/sda:$PYTHONPATH" >> ~/.bashrc

# Create necessary directories
RUN mkdir -p /workspace/data/{inputs,outputs,raw,processed} && \
    mkdir -p /workspace/results/{models,figures,logs}

# Set environment variables
ENV CONDA_DEFAULT_ENV=sda
ENV PATH=/opt/conda/envs/sda/bin:$PATH
ENV PYTHONPATH=/workspace:/workspace/sda

# Expose ports
EXPOSE 8888 6006 8000

# Default command
CMD ["conda", "run", "-n", "sda", "bash"]