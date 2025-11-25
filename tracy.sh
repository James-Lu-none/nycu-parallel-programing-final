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
)


renders=(
    pthread_simd
    pthread
    serial
    cuda
)

sleep_time=30

if [ ! -d "tracy" ]; then
    mkdir tracy
fi
if [ ! -d "eval" ]; then
    mkdir eval
fi

for ((i=2; i<=10; i+=2)); do
    for acc in "${accs[@]}"; do
        cd build
        rm -rf *
        cmake -DACCEL_VARIANT=$acc -DRENDER_VARIANT=serial ..
        make
        ./N_body assets/random_1000.txt $i &
        PID=$!
        cd ..
        ./tracy-capture.exe -a localhost -o ./tracy/accelerations_${acc}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        ./tracy-csvexport.exe ./tracy/accelerations_${acc}_$i.tracy > ./eval/accelerations_${acc}_$i.csv
    done

    for render in "${renders[@]}"; do
        cd build
        rm -rf *
        cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=$render ..
        make
        ./N_body assets/random_1000.txt $i &
        PID=$!
        cd ..
        ./tracy-capture.exe -a localhost -o ./tracy/render_${render}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        ./tracy-csvexport.exe ./tracy/render_${render}_$i.tracy > ./eval/render_${render}_$i.csv
    done
done

python eval.py