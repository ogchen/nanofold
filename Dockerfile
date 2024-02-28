FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
        cmake \
        build-essential \
        git \
        python3.12 \
        python3-pip \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/hhsuite/ \
    && git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hhsuite \
    && mkdir /tmp/hhsuite/build \
    && cd /tmp/hhsuite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 \
    && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && rm -rf /tmp/hhsuite

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
