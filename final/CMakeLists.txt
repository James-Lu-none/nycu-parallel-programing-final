cmake_minimum_required(VERSION 3.10)
project(N_body LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/tracy/public/tracy)
include_directories(${PROJECT_SOURCE_DIR}/include/tracy/public)

add_executable(N_body src/N_body.cpp)

target_link_libraries(N_body
    PUBLIC
    TracyClient
    ${SDL2_LIBRARIES}
    accelerations
    m
)

add_library(TracyClient STATIC ${PROJECT_SOURCE_DIR}/include/tracy/public/TracyClient.cpp)
target_compile_definitions(TracyClient PUBLIC TRACY_ENABLE=1)

add_subdirectory(src/accelerations)

target_include_directories(N_body PUBLIC ${SDL2_INCLUDE_DIRS})
target_compile_options(N_body PUBLIC -Wall -O2 -pg -g)
