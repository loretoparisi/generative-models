#
#
# @author loretoparisi at gmail dot com
#
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL Loreto Parisi (loretoparisi@gmail.com)

ENV DEBIAN_FRONTEND=noninteractive
ENV ENV PIP_ROOT_USER_ACTION=ignore
ENV TRANSFORMERS_CACHE=/opt/program/checkpoints
ENV TORCH_HOME=/opt/program/checkpoints

# install deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git

# fix: ImportError: libGL.so.1: cannot open shared object file: No such file or directory    
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Project setup
WORKDIR /opt/program

COPY requirements/pt2.txt /opt/program/requirements.txt

# install requirements in order
RUN pip install -r /opt/program/requirements.txt

# copy src
COPY . /opt/program/

# default cmd executes inference
CMD ["bash"]
