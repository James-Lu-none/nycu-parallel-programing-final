#!/bin/bash
# if build directory doesn't exist, create it
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
rm -rf *
cmake -DACCEL_VARIANT=serial_simd -DRENDER_VARIANT=pthread ..
# cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=pthread ..
make
./N_body