#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
#

# LP: this should be optional on mac M1 only
export DOCKER_BUILDKIT=0                                                                                                                                                    
export COMPOSE_DOCKER_CLI_BUILD=0
export DOCKER_DEFAULT_PLATFORM=linux/amd64

IMAGE_NAME=$1

[ -z "$IMAGE_NAME" ] && { echo "Usage: $0 IMAGE_NAME"; exit 1; }

if [ ! -f Dockerfile ]; then
    echo "Dockerfile file not found!"; exit 1;
fi

echo "Building \"${IMAGE_NAME}\" ..."
docker build -t ${IMAGE_NAME} -f Dockerfile .
