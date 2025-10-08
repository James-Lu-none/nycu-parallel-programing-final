set(ACCEL_VARIANT "serial" CACHE STRING "Acceleration implementation variant")

if(ACCEL_VARIANT STREQUAL "serial")
    add_library(accelerations accelerations_serial.cpp)
elseif(ACCEL_VARIANT STREQUAL "pthread_balanced")
    add_library(accelerations accelerations_pthread_balanced.cpp)
elseif(ACCEL_VARIANT STREQUAL "pthread_pair")
    add_library(accelerations accelerations_pthread_pair.cpp)
elseif(ACCEL_VARIANT STREQUAL "pthread_pair_persist")
    add_library(accelerations accelerations_pthread_pair_persist.cpp)
    target_compile_definitions(accelerations PUBLIC INIT_REQUIRED=1)
else()
    message(FATAL_ERROR "Unknown accelerations variant: ${ACCEL_VARIANT}")
endif()

target_include_directories(accelerations PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(accelerations PUBLIC TracyClient)
target_compile_definitions(accelerations PUBLIC TRACY_ENABLE=1)
