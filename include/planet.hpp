#ifndef PLANET_HPP
#define PLANET_HPP
#include "config.hpp"
#include "vec3.hpp"

typedef struct
{
    point3 pos;
    vec3 vel;
    vec3 acc;
    double mass;
    double r;
} Planet;

typedef struct
{
    point3 pos[TRAIL_BUF];
    int head;
    int size;
} Trail;

#endif