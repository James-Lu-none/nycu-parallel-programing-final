#!/bin/bash

accs=(
    pthread_blocked
    pthread_interleaved
    pthread_mutex_blocked
    pthread_mutex_interleaved
    pthread_mutex_simd_blocked
    pthread_mutex_simd_interleaved
    pthread_simd_blocked
    pthread_simd_interleaved
    serial_simd
    serial
    cuda_blocked
    cuda_interleaved
)


renders=(
    serial
    serial_simd
    pthread
    pthread_simd
    pthread_mutex
    pthread_mutex_simd
    cuda
)

sleep_time=10

if [ ! -d "tracy" ]; then
    mkdir tracy
fi
if [ ! -d "eval" ]; then
    mkdir eval
fi

for acc in "${accs[@]}"; do
    cd build || exit
    rm -rf *
    cmake -DACCEL_VARIANT=$acc -DRENDER_VARIANT=serial ..
    make
    cd ..
    for ((i=2; i<=10; i+=2)); do
        ./build/N_body ./assets/random_1000.txt $i &
        PID=$!
        while ! nc -z localhost 8086; do
            echo "Waiting for Tracy server to start..."
            sleep 1
        done
        sleep 1
        ./tracy-capture -a localhost -o ./tracy/accelerations_${acc}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        ./tracy-csvexport ./tracy/accelerations_${acc}_$i.tracy > ./eval/accelerations_${acc}_$i.csv
    done
done

for render in "${renders[@]}"; do
    cd build || exit
    rm -rf *
    cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=$render ..
    make
    cd ..
    for ((i=2; i<=10; i+=2)); do
        ./build/N_body ./assets/random_1000.txt $i &
        PID=$!
        while ! nc -z localhost 8086; do
            echo "Waiting for Tracy server to start..."
            sleep 1
        done
        sleep 1
        ./tracy-capture -a localhost -o ./tracy/render_${render}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        ./tracy-csvexport ./tracy/render_${render}_$i.tracy > ./eval/render_${render}_$i.csv
    done
done

python eval.py