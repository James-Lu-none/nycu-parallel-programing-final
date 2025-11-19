#pragma once
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include "canvas.hpp"
#include "config.hpp"

void render(
    canvas &buf, 
    const point3 &camera_center,
    const vec3 &pixel00_loc,
    const vec3 &pixel_delta_u,
    const vec3 &pixel_delta_v,
    const Planet* bodies,
    const Trail* trails = nullptr
);

color get_ray_color(const ray &r, const Planet* bodies, const Trail* trails);
double hit_planet(const Planet &p, const ray &r);
bool hit_trail(const Trail &t, const ray &r);

void trail_push(Trail *t, vec3 pos);
void recenter(Planet b[]);