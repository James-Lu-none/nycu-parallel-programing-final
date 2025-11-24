#pragma once
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include "canvas.hpp"
#include "camera.hpp"
#include "config.hpp"

void render(
    uint32_t *pixels,
    const Camera &camera,
    const PlanetsSoA &bodies,
    const Trail* trails = nullptr
);

color get_ray_color(const ray &r, const PlanetsSoA& bodies, const Trail *trails);
color get_ray_color_simd(const ray &r, const PlanetsSoA& bodies, const Trail *trails);

float hit_planet(vec3 pos, float r, const ray &ray_obj);
float hit_trail(const Trail &t, const ray &r);

void trail_push(Trail *t, vec3 pos);
void recenter(vector<Planet>& bodies);