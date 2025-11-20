add_library(camera
    camera.cpp
)

target_include_directories(camera PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(camera PUBLIC
    TracyClient
)

target_compile_definitions(camera PUBLIC TRACY_ENABLE=1)