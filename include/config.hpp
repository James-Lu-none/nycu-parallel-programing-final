#pragma once

#include <cstddef>
#include <cstdint>

namespace config {

inline constexpr int WIDTH = 1200;
inline constexpr int HEIGHT = 800;
inline constexpr double MIN_DIST = 1.5;
inline constexpr double G = 10000.0;
inline constexpr double EPSILON = 1e-6;
inline constexpr int NUM_THREADS = 4;
inline constexpr double VIEW_Z = 100.0;
inline constexpr std::size_t DEFAULT_BODY_COUNT = 3;
inline constexpr std::size_t DEFAULT_TRAIL_LENGTH = 100;

struct RuntimeOptions {
    std::size_t body_count = DEFAULT_BODY_COUNT;
    std::size_t trail_length = DEFAULT_TRAIL_LENGTH;
    bool start_in_ray_tracing_mode = false;
};

RuntimeOptions parse_arguments(int argc, char **argv);

} // namespace config

