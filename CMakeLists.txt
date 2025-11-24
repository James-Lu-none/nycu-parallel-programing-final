cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")

project(N_body LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(-Wall -Wextra -O3 -mavx2 -mfma)

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

# add_library(project_common INTERFACE)
# target_compile_options(project_common INTERFACE
#     -Wall -O3 -pg -g
# )
# target_compile_definitions(project_common INTERFACE TRACY_ENABLE=1)
# target_link_libraries(project_common INTERFACE TracyClient)

add_subdirectory(src/accelerations)
add_subdirectory(src/render)
add_subdirectory(src/integrator)
add_subdirectory(src/planet)
add_subdirectory(src/camera)

target_include_directories(N_body PUBLIC ${SDL2_INCLUDE_DIRS})
