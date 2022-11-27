# cuda version
ARG CUDA_VERSION=11.7.1-base-ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}

# python and pytorch versions
ARG PYTHON_VERSIONS='python3.8 python3-pip'
ARG PYTORCH_INSTALL='--pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117'
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y --no-install-recommends -y wget libpng-dev net-tools ${PYTHON_VERSIONS} && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	apt-get clean && \
	rm -rf /tmp/* /var/tmp/*

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir ${PYTORCH_INSTALL}

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip3 pip3 install -r requirements.txt

# cache pretrained networks
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ \
  && wget --progress=bar:force:noscroll https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_small_patch16_224_mlp_im_1k_224.pth -P /root/.cache/torch/hub/checkpoints/
