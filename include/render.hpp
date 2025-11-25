#pragma once
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include "canvas.hpp"
#include "camera.hpp"
#include "config.hpp"

void render(
    void *buf,
    const Camera &camera,
    const vector<Planet>& bodies,
    const Trail* trails = nullptr
);

void init_render_workers();
void destroy_render_workers();

color get_ray_color(const ray &r, const vector<Planet>& bodies, const Trail *trails);
color get_ray_color_simd(const ray &r, const vector<Planet>& bodies, const Trail *trails);

float hit_planet(const Planet &p, const ray &r);
float hit_trail(const Trail &t, const ray &r);

void trail_push(Trail *t, vec3 pos);
void recenter(vector<Planet>& bodies);