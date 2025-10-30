# NVIDIA Transformers container for Thor GPU
# Using NVIDIA AI-IOT transformers base image with CUDA 13.0 support
FROM ghcr.io/nvidia-ai-iot/transformers:r38.2.arm64-sbsa-cu130-24.04

# Set timezone and non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install development tools and OpenGL libraries for cv2
RUN apt-get update && apt-get install -y \
    nano \
    git \
    x11-apps \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies for the project
RUN pip install --no-cache-dir \
    'numpy<2' \
    einops \
    efficientnet_pytorch \
    PyYAML \
    Pillow \
    prettytable \
    matplotlib \
    scipy \
    diffusers \
    opencv-python \
    vit_pytorch \
    wandb \
    warmup_scheduler \
    pyzmq \
    gdown

# Set working directory
WORKDIR /workspace

# Copy diffusion_policy and train packages
COPY workspace/diffusion_policy ./diffusion_policy
COPY workspace/train ./train

# Create __init__.py files
RUN find ./diffusion_policy -type d -exec touch {}/__init__.py \; && \
    find ./train -type d -exec touch {}/__init__.py \;

# Install project packages
RUN pip install --no-cache-dir ./diffusion_policy && \
    pip install --no-cache-dir ./train

# Copy important data files to both possible locations
# Check where vint_train is installed and copy data files there
RUN VINT_PATH=$(python3 -c "import vint_train; import os; print(os.path.dirname(vint_train.__file__))" 2>/dev/null || echo "/opt/venv/lib/python3.12/site-packages/vint_train") && \
    mkdir -p ${VINT_PATH}/data && \
    cp ./train/vint_train/data/data_config.yaml ${VINT_PATH}/data/ && \
    cp ./train/vint_train/data/data_utils.py ${VINT_PATH}/data/ && \
    cp ./train/vint_train/data/vint_dataset.py ${VINT_PATH}/data/ && \
    echo "Copied data files to ${VINT_PATH}/data"

# Clean up
RUN rm -rf ./diffusion_policy ./train

CMD ["/bin/bash"]
