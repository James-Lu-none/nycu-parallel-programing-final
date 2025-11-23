set(RENDER_VARIANT "serial" CACHE STRING "render implementation variant")


if(RENDER_VARIANT STREQUAL "serial")
    add_library(render render_serial.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread")
    add_library(render render_pthread.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_simd")
    add_library(render render_pthread_simd.cpp)
else()
    message(FATAL_ERROR "Unknown render variant: ${RENDER_VARIANT}")
endif()

target_sources(render PUBLIC ${PROJECT_SOURCE_DIR}/src/render/common.cpp)

target_include_directories(render PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(render PUBLIC TracyClient)
target_compile_definitions(render PUBLIC TRACY_ENABLE=1)
