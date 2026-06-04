#!/bin/bash
# Usage: ./build.sh [-gpu] [-gpu-reverse]
#   (no flags)    CPU-only build
#   -gpu          Enable troy GPU HE backend (TRIPLE_GPU=ON)
#   -gpu-reverse  Enable GPU with reversed encryption roles (TRIPLE_GPU=ON TRIPLE_GPU_REVERSE=ON)
#                 In reverse mode the weight-holder encrypts weights instead of the image-holder
#                 encrypting inputs — can be faster when weights are smaller than the input tensor.

BUILD_DIR=build
FERRET_DIR="data"
TRIPLE_GPU=OFF
TRIPLE_GPU_REVERSE=OFF

while [[ $# -gt 0 ]]; do
    case "$1" in
        -gpu)         TRIPLE_GPU=ON; shift ;;
        -gpu-reverse) TRIPLE_GPU=ON; TRIPLE_GPU_REVERSE=ON; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d $FERRET_DIR ]]; then
    mkdir $FERRET_DIR
else
    rm -f $FERRET_DIR/*
fi

if [[ $TRIPLE_GPU == ON ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DUSE_APPROX_RESHARE=OFF \
        -DTRIPLE_VERIFY=OFF -DTRIPLE_COLOR=OFF -DTRIPLE_ZERO=ON \
        -DTRIPLE_GPU=ON \
        -DTRIPLE_GPU_REVERSE=${TRIPLE_GPU_REVERSE}
else
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DUSE_APPROX_RESHARE=OFF \
        -DTRIPLE_VERIFY=OFF -DTRIPLE_COLOR=OFF -DTRIPLE_ZERO=ON
fi

cmake --build build -j
