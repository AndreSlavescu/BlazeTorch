# BlazeTorch

A toy torch jit custom compiler to accelerate models.

## Build

The project gets built with CMake by running:
```
chmod +x build.sh
./build.sh
```

## Running Tests

All tests are hosted in the tests directory. To test the optimized compiler and see the end to end speedup for BERT, run the following:
```
python3 tests/run_bert.py
```

adapted from:

https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch

