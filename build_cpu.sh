#!/bin/bash

BUILD_TYPE=Release
BUILD_DIR=build
FERRET_DIR="data"

if [[ ! -d $BUILD_DIR ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DTRIPLE_VERIFY=OFF \
        -DTRIPLE_COLOR=OFF -DUSE_APPROX_RESHARE=OFF -DTRIPLE_ZERO=ON \
        -DTRIPLE_GPU=OFF \
        -DCMAKE_CXX_COMPILER=g++
fi

if [[ ! -d $FERRET_DIR ]]; then
    mkdir $FERRET_DIR
else
    rm -f $FERRET_DIR/*
fi

cmake --build build -j
