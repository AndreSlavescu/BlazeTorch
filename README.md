# BlazeTorch

A toy torch jit custom compiler to accelerate models.

## Clone and Build Dependencies
```
git clone https://github.com/AndreSlavescu/BlazeTorch
git submodule update --init --recursive
```

## Build

The project gets built with CMake by running:
```
chmod +x build.sh
./build.sh
```

## Running Tests

All tests are hosted in the tests directory. To test the optimized compiler and see the end to end speedup for BERT, run the following:
```
python3 run_bert.py
```

To generate the graph trace for the optimized and base torch eager execution profiles, run the following:
```
python3 run_bert.py --generate_trace
```

traces will be hosted in a built ```traces/``` directory within ```tests/```

adapted and extended from:

https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch

