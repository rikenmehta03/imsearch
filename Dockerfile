# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    GIT_CLONE="git clone --depth 10" && \
    \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        bash \
        nano \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \
    \
    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \
    \
# ==================================================================
# python
# ------------------------------------------------------------------
    \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \
   \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
   \
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre torch torchvision -f \
        https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html \
        && \
   \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
   \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# install nmslib
# ------------------------------------------------------------------
RUN pip install --no-binary :all: nmslib

# ==================================================================
# install imsearch
# ------------------------------------------------------------------
WORKDIR /imsearch
COPY . /imsearch

# ==================================================================
# test imsearch
# ------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install nose2 && \
    nose2

RUN pip install .

CMD ["/bin/bash"]
