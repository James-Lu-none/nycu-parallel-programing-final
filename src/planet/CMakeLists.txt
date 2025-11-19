add_library(planet
    planet.cpp
)

target_include_directories(planet PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(planet PUBLIC
    TracyClient
)

target_compile_definitions(planet PUBLIC TRACY_ENABLE=1)