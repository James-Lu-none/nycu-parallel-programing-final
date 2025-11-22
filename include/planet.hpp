#ifndef PLANET_HPP
#define PLANET_HPP
#include "config.hpp"
#include "vec3.hpp"
#include "canvas.hpp"

typedef struct
{
    point3 pos;
    vec3 vel;
    vec3 acc;
    float mass;
    float r;
    color col;
} Planet;

typedef struct
{
    point3 pos[TRAIL_BUF];
    int head;
    int size;
} Trail;

uint64_t load_planets_from_file(const char *filename, Planet *b);
vec3 get_center_of_mass(Planet b[]);
void trail_push(Trail *t, vec3 pos);

#endif