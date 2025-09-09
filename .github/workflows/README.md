# GitHub Workflows Documentation

## Available Workflows

This repository contains several GitHub workflows for building Docker images targeting different platforms and CUDA versions:

### Individual Workflows

1. **`docker.cpu.yml`** - CPU-only Docker image
   - **Triggers**: Changes to `docker/dockerfile.cpu` or workflow file
   - **Image Tag**: `cpu-torch`
   - **Use Case**: CPU-only execution, development without GPU

2. **`docker.gpu.cu124.yml`** - GPU Docker image with CUDA 12.4
   - **Triggers**: Changes to `docker/dockerfile.gpu.cu124` or workflow file
   - **Image Tag**: `gpu-torch-cuda12.4`
   - **Use Case**: RTX 3080/3090/4080/4090 and older GPUs

3. **`docker.gpu.cu129.yml`** - GPU Docker image with CUDA 12.9
   - **Triggers**: Changes to `docker/dockerfile.gpu.cu129` or workflow file
   - **Image Tag**: `gpu-torch-cuda12.9-rtx5090`
   - **Use Case**: RTX 5090 and latest GPUs with Blackwell architecture

4. **`docker.nano.yml`** - Jetson Nano ARM64 Docker image
   - **Triggers**: Changes to `docker/dockerfile.nano` or workflow file
   - **Image Tag**: `l4t-torch-36.4.0`
   - **Platform**: `linux/arm64`
   - **Use Case**: NVIDIA Jetson Nano devices

## Workflow Structure

Each workflow follows the same basic structure:

1. **Checkout**: Repository checkout with `actions/checkout@v4`
2. **Docker Meta**: Image metadata generation with `docker/metadata-action@v5`
3. **Login**: Docker Hub authentication using secrets
4. **Build & Push**: Image build and push with `docker/build-push-action@v6`

## Required Secrets

The workflows require the following GitHub repository secrets:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_ACCESS_TOKEN`: Your Docker Hub access token

## Image Tags

| Workflow | Image Tag | Target Platform |
|----------|-----------|----------------|
| CPU | `cpu-torch` | x86_64 CPU |
| GPU CUDA 12.4 | `gpu-torch-cuda12.4` | x86_64 + GPU (older) |
| GPU CUDA 12.9 | `gpu-torch-cuda12.9-rtx5090` | x86_64 + RTX 5090 |
| Jetson Nano | `l4t-torch-36.4.0` | ARM64 (Jetson) |

## Usage in Compose Files

The workflows build images that correspond to the compose files:

- `compose.cpu.yml` → `cpu-torch`
- `compose.gpu.cu124.yml` → `gpu-torch-cuda12.4`
- `compose.gpu.cu129.yml` → `gpu-torch-cuda12.9-rtx5090`
- `compose.nano.yml` → `l4t-torch-36.4.0`
