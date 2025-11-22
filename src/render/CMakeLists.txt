set(RENDER_VARIANT "serial" CACHE STRING "render implementation variant")


if(RENDER_VARIANT STREQUAL "serial")
    add_library(render render_serial.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread")
    add_library(render render_pthread.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_simd")
    add_library(render render_pthread_simd.cpp)
elseif(RENDER_VARIANT STREQUAL "cuda")
    enable_language(CUDA)
    add_library(render render_cuda.cu)
    target_compile_options(render PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
else()
    message(FATAL_ERROR "Unknown render variant: ${RENDER_VARIANT}")
endif()

if(NOT RENDER_VARIANT STREQUAL "cuda")
    target_sources(render PUBLIC ${PROJECT_SOURCE_DIR}/src/render/common.cpp)
endif()

target_include_directories(render PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(render PUBLIC TracyClient)

if(NOT RENDER_VARIANT STREQUAL "cuda")
    target_compile_definitions(render PUBLIC TRACY_ENABLE=1)
    target_compile_options(render PUBLIC -O3 -mavx2 -mfma)
endif()

