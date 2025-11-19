add_library(integrator
    step_leapfrog.cpp
)

target_include_directories(integrator PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(integrator PUBLIC
    accelerations
    TracyClient
)

target_compile_definitions(integrator PUBLIC TRACY_ENABLE=1)