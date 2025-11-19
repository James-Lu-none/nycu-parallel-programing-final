#pragma once
#include <cstdint>
#include "config.hpp"

struct color
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

struct canvas
{
    color pixels[WIDTH * HEIGHT];
};
