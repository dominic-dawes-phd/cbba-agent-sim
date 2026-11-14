# The cluster runtime (docs/design/cluster.md): llama.cpp's official CUDA server image, pinned to
# the build the laptop runs (b12211), plus Python 3.12, PyYAML, and git.
# The code is not copied in: the cluster bind-mounts a git checkout of this repo at /work, so a
# code change needs only `git pull`. Rebuild the image only to change llama.cpp or Python.
# The base needs an NVIDIA driver that supports CUDA 12.8 (driver 570 or newer).
ARG LLAMA_TAG=server-cuda-b12211
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_TAG}

# Ubuntu 24.04's python3 is 3.12; python3-yaml is the sim's only runtime dependency; git lets
# run.json record the commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-yaml git \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/app:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
# The base image starts llama-server; this image runs whatever command it is given.
ENTRYPOINT []
CMD ["bash"]
