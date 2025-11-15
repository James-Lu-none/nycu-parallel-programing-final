set(ACCEL_VARIANT "serial" CACHE STRING "Acceleration implementation variant")

if(ACCEL_VARIANT STREQUAL "serial")
    add_library(accelerations accelerations_serial.cpp)
elif(ACCEL_VARIANT STREQUAL "pthread_balanced")
    add_library(accelerations accelerations_pthread_balanced.cpp)
elif(ACCEL_VARIANT STREQUAL "pthread_pair")
    add_library(accelerations accelerations_pthread_pair.cpp)
elif(ACCEL_VARIANT STREQUAL "pthread_pair_interleaved")
    add_library(accelerations accelerations_pthread_pair_interleaved.cpp)
elif(ACCEL_VARIANT STREQUAL "pthread_pair_persist")
    add_library(accelerations accelerations_pthread_pair_persist.cpp)
elif(ACCEL_VARIANT STREQUAL "simd")
    add_library(accelerations accelerations_simd.cpp)
    target_compile_options(accelerations PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2 -mfma>
    )
elif(ACCEL_VARIANT STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(accelerations accelerations_cuda.cu)
    set_target_properties(accelerations PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(accelerations PUBLIC CUDA::cudart)
else()
    message(FATAL_ERROR "Unknown accelerations variant: ${ACCEL_VARIANT}")
endif()

target_include_directories(accelerations PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(accelerations PUBLIC TracyClient)
target_compile_definitions(accelerations PUBLIC TRACY_ENABLE=1)
