# BlazeTorch

A toy torch jit custom compiler to accelerate models.

## Clone and Build Dependencies
Before building, make sure you have cmake installed. Tested on cmake version 3.29.3.

```
git clone https://github.com/AndreSlavescu/BlazeTorch
git submodule update --init --recursive

# build flatbuffers with appropriate version (23 major version and 3 minor version)
cd submodules/nvfuser/third_party/flatbuffers
git fetch --all
git checkout v23.3.3
mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/local ..
make -j$(nproc)
make install
cd ../../../../..
```

## Build

The project gets built with CMake by running:
```
chmod +x build.sh
./build.sh
```

## Running Tests

All tests are hosted in the tests directory. To test the optimized compiler and see the following examples:

Testing BERT:
```
python3 tests/run_bert.py
```

To generate the graph trace for the optimized and base torch eager execution profiles, run the following:
```
python3 tests/run_bert.py --generate_trace
```

Testing Llama:

To successfully test the Llama-7b model, you need to create a `.env` file and paste in your huggingface access token as follows:

```
# .env
HF_TOKEN="your token"
```

Then export the .env file stored in your project root.

```
python3 tests/run_llama.py --generate_trace
```


traces will be hosted in a built ```tests/traces/```

adapted and extended from:

https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch

