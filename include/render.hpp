#pragma once
#include "planet.hpp"
#include "config.hpp"

struct color
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct canvas
{
    color pixels[WIDTH * HEIGHT];
};

void recenter(Planet b[]);
void fill_circle(SDL_Surface *surf, int cx, int cy, int cz, int rad, Uint32 col);
void trail_push(Trail *t, int x, int y, int z);
void trail_draw(SDL_Surface *surf, const Trail *t, Uint32 col);