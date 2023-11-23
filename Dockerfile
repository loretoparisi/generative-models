#
#
# @author loretoparisi at gmail dot com
#
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0

LABEL Loreto Parisi (loretoparisi@gmail.com)

# install deps
RUN apt-get update && apt-get install -y \
    build-essential \
    nano 

# fix: ImportError: libGL.so.1: cannot open shared object file: No such file or directory    
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Project setup
WORKDIR /opt/program

COPY requirements/pt2.txt /opt/program/requirements.txt

# install requirements in order
RUN pip install -r /opt/program/requirements.txt
RUN pip install torch>=2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# copy src
COPY . /opt/program/

# default cmd executes inference
CMD ["bash"]
