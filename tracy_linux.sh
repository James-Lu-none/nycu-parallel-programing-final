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
    pthread_simd
    pthread
    serial
    cuda
)

sleep_time=30

# Function to find tracy tools
find_tool() {
    local tool_name=$1
    local tool_path=""

    # Check system PATH
    if command -v "$tool_name" &> /dev/null; then
        echo "$tool_name"
        return 0
    fi

    # Check common CMake build locations
    if [ -f "$HOME/Desktop/tracy/$tool_name/build/$tool_name" ]; then
        echo "$HOME/Desktop/tracy/$tool_name/build/$tool_name"
        return 0
    fi

    if [ -f "$HOME/Desktop/tracy/$tool_name/build/tracy-$tool_name" ]; then
        echo "$HOME/Desktop/tracy/$tool_name/build/tracy-$tool_name"
        return 0
    fi

    # Check current directory
    if [ -f "./$tool_name" ]; then
        echo "./$tool_name"
        return 0
    fi

    return 1
}

# Find tools
TRACY_CAPTURE=$(find_tool "capture")
if [ -z "$TRACY_CAPTURE" ]; then
    # Try looking for tracy-capture
    TRACY_CAPTURE=$(find_tool "tracy-capture")
fi

TRACY_CSVEXPORT=$(find_tool "csvexport")
if [ -z "$TRACY_CSVEXPORT" ]; then
    TRACY_CSVEXPORT=$(find_tool "tracy-csvexport")
fi

# Check if tools were found
if [ -z "$TRACY_CAPTURE" ] || [ -z "$TRACY_CSVEXPORT" ]; then
    echo "Error: Tracy tools not found."
    echo "Please build them using the following commands:"
    echo ""
    if [ -z "$TRACY_CAPTURE" ]; then
        echo "  cmake -B ~/Desktop/tracy/capture/build -S ~/Desktop/tracy/capture -DCMAKE_BUILD_TYPE=Release"
        echo "  cmake --build ~/Desktop/tracy/capture/build --parallel"
    fi
    echo ""
    if [ -z "$TRACY_CSVEXPORT" ]; then
        echo "  cmake -B ~/Desktop/tracy/csvexport/build -S ~/Desktop/tracy/csvexport -DCMAKE_BUILD_TYPE=Release"
        echo "  cmake --build ~/Desktop/tracy/csvexport/build --parallel"
    fi
    echo ""
    exit 1
fi

echo "Using capture tool: $TRACY_CAPTURE"
echo "Using csvexport tool: $TRACY_CSVEXPORT"

if [ ! -d "tracy" ]; then
    mkdir tracy
fi
if [ ! -d "eval" ]; then
    mkdir eval
fi

for ((i=2; i<=10; i+=2)); do
    for acc in "${accs[@]}"; do
        cd build || exit
        rm -rf *
        cmake -DACCEL_VARIANT=$acc -DRENDER_VARIANT=serial ..
        make
        ./N_body ../assets/random_1000.txt $i &
        PID=$!
        cd ..
        $TRACY_CAPTURE -a localhost -o ./tracy/accelerations_${acc}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        $TRACY_CSVEXPORT ./tracy/accelerations_${acc}_$i.tracy > ./eval/accelerations_${acc}_$i.csv
    done

    for render in "${renders[@]}"; do
        cd build || exit
        rm -rf *
        cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=$render ..
        make
        ./N_body ../assets/random_1000.txt $i &
        PID=$!
        cd ..
        $TRACY_CAPTURE -a localhost -o ./tracy/render_${render}_$i.tracy &
        sleep $sleep_time
        kill $PID
        sleep 0.5
        $TRACY_CSVEXPORT ./tracy/render_${render}_$i.tracy > ./eval/render_${render}_$i.csv
    done
done

python eval.py