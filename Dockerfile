FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates unzip \
    libyaml-cpp-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- LibTorch (CPU) ----
RUN curl -L "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip" -o /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV CMAKE_PREFIX_PATH=/opt/libtorch

# ---- Non-root user (optional but nice) ----
RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /work
