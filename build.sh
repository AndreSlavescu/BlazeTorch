#!/bin/bash
build_dir="build"

if [ -d "$build_dir" ]; then
    rm -rf "$build_dir"
fi

mkdir "$build_dir" && cd "$build_dir"
cmake ..
make -j 8
