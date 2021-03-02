#############################
# Info about the base image #
#############################

# Docker Hub:
# https://hub.docker.com/layers/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-runtime/images/sha256-9c3aa4653f6fb6590acf7f49115735be3c3272f4fa79e5da7c96a2c901631352?context=explore
# 
# Dockerfile:
# https://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# ARG BASE_IMAGE=ubuntu:18.04
# ARG PYTHON_VERSION=3.7
# 
# packages in environment at /opt/conda:
#
# Name                    Version                   Build  Channel
# _libgcc_mutex             0.1                        main  
# backcall                  0.2.0                      py_0  
# beautifulsoup4            4.9.1                    py37_0  
# blas                      1.0                         mkl  
# bzip2                     1.0.8                h7b6447c_0  
# ca-certificates           2020.1.1                      0  
# certifi                   2020.4.5.2               py37_0  
# cffi                      1.14.0           py37he30daa8_1  
# chardet                   3.0.4                 py37_1003  
# conda                     4.8.3                    py37_0  
# conda-build               3.18.11                  py37_0  
# conda-package-handling    1.6.1            py37h7b6447c_0  
# cryptography              2.9.2            py37h1ba5d50_0  
# cudatoolkit               10.1.243             h6bb024c_0  
# decorator                 4.4.2                      py_0  
# filelock                  3.0.12                     py_0  
# freetype                  2.9.1                h8a8886c_1  
# glob2                     0.7                        py_0  
# icu                       58.2                 he6710b0_3  
# idna                      2.9                        py_1  
# intel-openmp              2020.1                      217  
# ipython                   7.15.0                   py37_0  
# ipython_genutils          0.2.0                    py37_0  
# jedi                      0.17.0                   py37_0  
# jinja2                    2.11.2                     py_0  
# jpeg                      9b                   h024ee3a_2  
# ld_impl_linux-64          2.33.1               h53a641e_7  
# libarchive                3.4.2                h62408e4_0  
# libedit                   3.1.20181209         hc058e9b_0  
# libffi                    3.3                  he6710b0_1  
# libgcc-ng                 9.1.0                hdf63c60_0  
# libgfortran-ng            7.3.0                hdf63c60_0  
# liblief                   0.10.1               he6710b0_0  
# libpng                    1.6.37               hbc83047_0  
# libstdcxx-ng              9.1.0                hdf63c60_0  
# libtiff                   4.1.0                h2733197_1  
# libxml2                   2.9.10               he19cac6_1  
# lz4-c                     1.9.2                he6710b0_0  
# markupsafe                1.1.1            py37h7b6447c_0  
# mkl                       2020.1                      217  
# mkl-service               2.3.0            py37he904b0f_0  
# mkl_fft                   1.1.0            py37h23d657b_0  
# mkl_random                1.1.1            py37h0573a6f_0  
# ncurses                   6.2                  he6710b0_1  
# ninja                     1.9.0            py37hfd86e86_0  
# numpy                     1.18.1           py37h4f9e942_0  
# numpy-base                1.18.1           py37hde5b4d6_1  
# olefile                   0.46                     py37_0  
# openssl                   1.1.1g               h7b6447c_0  
# parso                     0.7.0                      py_0  
# patchelf                  0.11                 he6710b0_0  
# pexpect                   4.8.0                    py37_0  
# pickleshare               0.7.5                    py37_0  
# pillow                    7.1.2            py37hb39fc2d_0  
# pip                       20.0.2                   py37_3  
# pkginfo                   1.5.0.1                  py37_0  
# prompt-toolkit            3.0.5                      py_0  
# psutil                    5.7.0            py37h7b6447c_0  
# ptyprocess                0.6.0                    py37_0  
# py-lief                   0.10.1           py37h403a769_0  
# pycosat                   0.6.3            py37h7b6447c_0  
# pycparser                 2.20                       py_0  
# pygments                  2.6.1                      py_0  
# pyopenssl                 19.1.0                   py37_0  
# pysocks                   1.7.1                    py37_0  
# python                    3.7.7                hcff3b4d_5  
# python-libarchive-c       2.9                        py_0  
# pytorch                   1.5.1           py3.7_cuda10.1.243_cudnn7.6.3_0    pytorch
# pytz                      2020.1                     py_0  
# pyyaml                    5.3.1            py37h7b6447c_0  
# readline                  8.0                  h7b6447c_0  
# requests                  2.23.0                   py37_0  
# ripgrep                   11.0.2               he32d670_0  
# ruamel_yaml               0.15.87          py37h7b6447c_0  
# setuptools                46.4.0                   py37_0  
# six                       1.14.0                   py37_0  
# soupsieve                 2.0.1                      py_0  
# sqlite                    3.31.1               h62c20be_1  
# tk                        8.6.8                hbc83047_0  
# torchvision               0.6.1                py37_cu101    pytorch
# tqdm                      4.46.0                     py_0  
# traitlets                 4.3.3                    py37_0  
# urllib3                   1.25.8                   py37_0  
# wcwidth                   0.2.4                      py_0  
# wheel                     0.34.2                   py37_0  
# xz                        5.2.5                h7b6447c_0  
# yaml                      0.1.7                had09818_2  
# zlib                      1.2.11               h7b6447c_3  
# zstd                      1.4.4                h0b5b093_3  

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update \
    && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
    apt-utils \
    bash-completion \
    ca-certificates \
    curl \
    ffmpeg \
    gcc \
    gettext \
    git \
    graphviz \
    htop \
    httpie \
    jq \
    libgraphviz-dev \
    libsm6 \
    libxext6\
    libxrender-dev \
    net-tools \
    openssh-client \
    openssh-server \
    rsync \
    sudo \
    tar \
    tmux \
    tree \
    unzip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

ARG LOCAL_ROOT_DIR=/workspace
COPY ./ ${LOCAL_ROOT_DIR}
WORKDIR ${LOCAL_ROOT_DIR}

# For PipelineX Users
RUN pip --no-cache-dir install -r requirements_optional.txt
RUN pip --no-cache-dir install pipelinex

# For PipelineX Contributors
# RUN pip --no-cache-dir install -r requirements_dev.txt
# RUN python setup.py develop

EXPOSE 22 4141 5000 5555 6006 8000 8080 8793 8888 8889
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["tail", "-f", "/dev/null"]
