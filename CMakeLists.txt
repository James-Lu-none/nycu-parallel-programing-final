cmake_minimum_required(VERSION 3.10)
project(N_body LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

include(FetchContent)

FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.12.2
)

FetchContent_MakeAvailable(tracy)

target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE=1)

add_executable(N_body src/N_body.cpp)

target_link_libraries(N_body
    PUBLIC
    TracyClient
    ${SDL2_LIBRARIES}
    accelerations
    render
    integrator
    planet
    camera
    m
)

add_subdirectory(src/accelerations)
add_subdirectory(src/render)
add_subdirectory(src/integrator)
add_subdirectory(src/planet)
add_subdirectory(src/camera)

target_include_directories(N_body PUBLIC ${SDL2_INCLUDE_DIRS})
target_compile_options(N_body PUBLIC -Wall -O2 -pg -g)
