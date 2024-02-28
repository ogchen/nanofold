FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
        cmake \
        build-essential \
        git \
        wget \
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
    && rm -rf /tmp/hhsuite

RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm -rf /tmp/miniconda.sh

ENV PATH="/opt/miniconda3/bin:/opt/hhsuite/bin:${PATH}"

RUN conda install -y -c conda-forge -c pytorch -c nvidia \
        python=3.12 \
        pip \
    && conda clean -afy

COPY . /app
WORKDIR /app

RUN pip3 install --no-cache-dir --default-timeout=1500 -r requirements.txt
