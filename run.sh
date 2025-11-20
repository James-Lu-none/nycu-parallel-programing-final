mkdir build
cd build
cmake -DACCEL_VARIANT=pthread_balanced -DRENDER_VARIANT=pthread ..
# cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=pthread ..
make
./N_body