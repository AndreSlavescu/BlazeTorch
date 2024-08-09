#!/bin/bash
build_dir="build"
nvfuser_dir="submodules/nvfuser"
nvfuser_build_marker="$nvfuser_dir/bin/CMakeFiles"

if [ ! -d "$nvfuser_build_marker" ]; then
    echo "Building nvfuser..."
    if [ -d "$build_dir" ]; then
        rm -rf "$build_dir"
    fi
    mkdir "$build_dir" && cd "$build_dir"
    cmake ..
    make -j 16 nvfuser_codegen
else
    echo "nvfuser is already built. Skipping nvfuser build."
    mkdir -p "$build_dir" && cd "$build_dir"
fi

make -j 16 blazetorch