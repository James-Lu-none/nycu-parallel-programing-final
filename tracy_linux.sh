#!/bin/bash
set -e

# Configuration
SLEEP_TIME=10
TRACY_CAPTURE="./tracy-capture"
TRACY_CSVEXPORT="./tracy-csvexport"
BUILD_DIR="build"
OUTPUT_DIR="eval"
TRACY_DIR="tracy"
ASSETS_DIR="./assets"
INPUT_FILE="random_1000.txt"

# Arrays
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

# Ensure directories exist

rm -r "$TRACY_DIR" || true
rm -r "$OUTPUT_DIR" || true

mkdir -p "$TRACY_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BUILD_DIR"

# Cleanup function to kill background processes on exit
cleanup() {
    # Kill any child processes of this script
    pkill -P $$ || true
}
trap cleanup EXIT

# Function to wait for port
wait_for_port() {
    local port=$1
    echo "Waiting for Tracy server on port $port..."
    while ! nc -z localhost "$port"; do
        sleep 1
    done
}

# Function to run benchmark
run_benchmark() {
    local type=$1      # ACCEL or RENDER
    local variant=$2   # Variant name
    local prefix=$3    # Output filename prefix

    echo "=== Benchmarking $type: $variant ==="

    # Build
    cd "$BUILD_DIR" || exit
    # Clean build directory safely
    find . -maxdepth 1 ! -name . -exec rm -rf {} +
    
    if [ "$type" == "ACCEL" ]; then
        cmake -DACCEL_VARIANT="$variant" -DRENDER_VARIANT=serial ..
    else
        cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT="$variant" ..
    fi
    make -j$(nproc)
    cd ..

    local threads="2 4 6 8 10"
    if  [[ "$variant" =~ cuda || "$variant" =~ serial ]]; then
        local threads="1"
    fi
    
    for i in $threads; do
        # Zero-pad the index
        local padded_i=$(printf "%02d" $i)
        
        echo "Running test with input size $i (Output: ${prefix}_${variant}_${padded_i})"

        ./"$BUILD_DIR"/N_body "$ASSETS_DIR/$INPUT_FILE" "$i" &
        local PID=$!

        wait_for_port 8086
        sleep 1

        "$TRACY_CAPTURE" -a localhost -o "$TRACY_DIR/${prefix}_${variant}_${padded_i}.tracy" &
        
        sleep "$SLEEP_TIME"
        
        # Kill N_body
        kill "$PID" || true
        wait "$PID" 2>/dev/null || true
        
        sleep 2
        "$TRACY_CSVEXPORT" -u -p "$TRACY_DIR/${prefix}_${variant}_${padded_i}.tracy" > "$OUTPUT_DIR/${prefix}_${variant}_${padded_i}.csv"
    done
}

# Run Acceleration Benchmarks
for acc in "${accs[@]}"; do
    run_benchmark "ACCEL" "$acc" "accelerations"
done

# Run Render Benchmarks
for render in "${renders[@]}"; do
    run_benchmark "RENDER" "$render" "render"
done

echo "Running evaluation script..."
python parse_eval.py eval parse_eval
python analysis.py