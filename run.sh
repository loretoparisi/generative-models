#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
#

IMAGE_NAME=$1
docker run --platform linux/amd64 -v $(pwd):/opt/program --gpus all --rm -it --shm-size 16G $IMAGE_NAME bash
