#pragma once
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"
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

void render(
    canvas &buf, 
    const point3 &camera_center,
    const vec3 &pixel00_loc,
    const vec3 &pixel_delta_u,
    const vec3 &pixel_delta_v
);

color get_ray_color(const ray &r);
void trail_push(Trail *t, vec3 pos);
void recenter(Planet b[]);