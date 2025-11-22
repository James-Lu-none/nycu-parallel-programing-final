#!/bin/bash
# if build directory doesn't exist, create it
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake -DACCEL_VARIANT=pthread_pair_persist -DRENDER_VARIANT=pthread ..
# cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=pthread ..
make
./N_body