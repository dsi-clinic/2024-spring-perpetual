#!/bin/bash

ARCH=$(uname -m)
case "$ARCH" in
    x86_64) echo "Setting platform to linux/amd64"
            export DOCKER_PLATFORM=linux/amd64
            ;;
    arm64) echo "Setting platform to linux/arm64"
           export DOCKER_PLATFORM=linux/arm64
           ;;
    *) echo "Unsupported architecture: $ARCH"
       exit 1
       ;;
esac
