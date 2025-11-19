set(RENDER_VARIANT "serial" CACHE STRING "render implementation variant")

if(RENDER_VARIANT STREQUAL "serial")
    add_library(render render_serial.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_balanced")
    add_library(render render_pthread_balanced.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_pair")
    add_library(render render_pthread_pair.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_pair_interleaved")
    add_library(render render_pthread_pair_interleaved.cpp)
elseif(RENDER_VARIANT STREQUAL "pthread_pair_persist")
    add_library(render render_pthread_pair_persist.cpp)
    target_compile_definitions(render PUBLIC INIT_REQUIRED=1)
else()
    message(FATAL_ERROR "Unknown render variant: ${RENDER_VARIANT}")
endif()

target_include_directories(render PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(render PUBLIC TracyClient)
target_compile_definitions(render PUBLIC TRACY_ENABLE=1)
