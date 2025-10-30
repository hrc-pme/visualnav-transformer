# NVIDIA PyTorch container with full Thor GPU support
# This container already includes PyTorch 2.8.0 with sm_110 (Thor) support
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set timezone and non-interactive modeA ROS distribution is a versioned set of ROS packages. These are akin to Linux distributions (e.g. Ubuntu). The purpose of the ROS distributions is to let developers work against a relatively stable codebase until they are ready to roll everything forward. Therefore once a distribution is released, we try to limit changes to bug fixes and non-breaking improvements for the core packages (every thing under ros-desktop-full). That generally applies to the whole community, but for “higher” level packages, the rules are less strict, and so it falls to the maintainers of a given package to avoid breaking changes.

List of Distributions
Below is a list of current and historic ROS 2 distributions. Rows in the table marked in green are the currently supported distributions.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install development tools
RUN apt-get update && apt-get install -y \
    nano \
    git \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies for the project
RUN pip install --no-cache-dir \
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
    pyzmq

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

# Copy important data files
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/vint_train/data && \
    cp ./train/vint_train/data/data_config.yaml /usr/local/lib/python3.12/dist-packages/vint_train/data/ && \
    cp ./train/vint_train/data/data_utils.py /usr/local/lib/python3.12/dist-packages/vint_train/data/ && \
    cp ./train/vint_train/data/vint_dataset.py /usr/local/lib/python3.12/dist-packages/vint_train/data/

# Clean up
RUN rm -rf ./diffusion_policy ./train

CMD ["/bin/bash"]
