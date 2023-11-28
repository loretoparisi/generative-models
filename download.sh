#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
#
echo "downloading stable-video-diffusion-img2vid..."

mkdir -p /opt/program/checkpoints
cd /opt/program/checkpoints
curl -o svd.safetensors "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true" -L
cd ..
