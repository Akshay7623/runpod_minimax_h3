# MiniMax H3 (FL2VA: text / image / first+last frame -> video with audio) on
# ComfyUI, packaged as a RunPod serverless worker. Every model file is baked
# into the image, so no network volume is needed.

# CUDA 13.0 on purpose: the int8_convrot checkpoints below are meant for a
# cu130 PyTorch, which needs NVIDIA driver >= 580. Set "Allowed CUDA Versions"
# to 13.0 on the endpoint so workers only land on hosts that have it.
ARG BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04

FROM ${BASE_IMAGE} AS base

# MiniMax H3 needs ComfyUI >= 0.30.0. Pinned so a rebuild gives the same image.
ARG COMFYUI_VERSION=v0.39.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_INPUT=1

# Globally disable pip and uv caches so they don't end up in the image
ENV PIP_NO_CACHE_DIR=1
ENV UV_NO_CACHE=1

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.12 \
  python3.12-venv \
  git \
  wget \
  ca-certificates \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ffmpeg \
  && ln -sf /usr/bin/python3.12 /usr/bin/python \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
  && ln -s /root/.local/bin/uv /usr/local/bin/uv \
  && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
  && uv venv --python /usr/bin/python3.12 /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install ComfyUI straight from its release tag. H3 runs on core nodes only, so
# comfy-cli and ComfyUI-Manager are not needed.
RUN git clone --depth 1 --branch "${COMFYUI_VERSION}" https://github.com/Comfy-Org/ComfyUI.git /comfyui

# torch goes in first, from the cu130 index: ComfyUI's requirements.txt asks for
# a bare `torch`, and installing it up front keeps the next step from picking a
# different build.
RUN uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

# ComfyUI's dependencies and the handler's, resolved together so a conflict
# fails the build instead of silently swapping a version ComfyUI needs.
# transformers 5.x / huggingface-hub 1.x have crashed ComfyUI at startup, and
# ComfyUI sets no upper bound on either, hence the caps.
RUN uv pip install -r /comfyui/requirements.txt \
  "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
  runpod==1.7.13 requests==2.32.4 websocket-client==1.8.0 boto3==1.40.21

# Build-time smoke test: start ComfyUI once on CPU so a broken dependency fails
# here, before 40 GB of models are downloaded, and not on a live worker.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Models, from the official ComfyUI repackage, pinned to one revision.
ARG H3_REPO=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6

# Diffusion model (21 GB) and the 8-step turbo LoRA (2 GB), fetched in parallel
RUN set -e; \
  fetch() { mkdir -p "/comfyui/models/$1" && wget -nv --tries=5 -O "/comfyui/models/$1/$2" "${H3_REPO}/$1/$2"; }; \
  fetch diffusion_models minimax_h3_fl2va_pruned_int8_convrot.safetensors & p1=$!; \
  fetch loras minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors & p2=$!; \
  wait "$p1"; wait "$p2"

# Text encoder (16 GB), video VAE (3 GB) and audio VAE (0.6 GB)
RUN set -e; \
  fetch() { mkdir -p "/comfyui/models/$1" && wget -nv --tries=5 -O "/comfyui/models/$1/$2" "${H3_REPO}/$1/$2"; }; \
  fetch text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors & p1=$!; \
  fetch vae minimax_h3_video_vae_int8_convrot.safetensors & p2=$!; \
  fetch vae minimax_h3_audio_vae_fp32.safetensors & p3=$!; \
  wait "$p1"; wait "$p2"; wait "$p3"

# Go back to the root
WORKDIR /

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Set the default command to run when starting the container
CMD ["/start.sh"]
