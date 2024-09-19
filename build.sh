#!/bin/bash
build_dir="build"
nvfuser_dir="submodules/nvfuser"
nvfuser_build_marker="$nvfuser_dir/bin/CMakeFiles"
build_with_nvfuser=false

# Parse arguments
for arg in "$@"
do
    if [ "$arg" = "build_with_nvfuser" ]; then
        build_with_nvfuser=true
    fi
done

if [ "$build_with_nvfuser" = true ]; then
    if [ ! -d "$nvfuser_build_marker" ]; then
        echo "Building nvfuser..."
        if [ -d "$build_dir" ]; then
            rm -rf "$build_dir"
        fi
        mkdir "$build_dir" && cd "$build_dir"
        cmake -DBUILD_NVFUSER=ON ..
        make -j 16 nvfuser_codegen
    else
        echo "nvfuser is already built. Skipping nvfuser build."
        mkdir -p "$build_dir" && cd "$build_dir"
    fi
else
    mkdir -p "$build_dir" && cd "$build_dir"
    cmake -DBUILD_NVFUSER=OFF ..
fi

make -j 16 blazetorch