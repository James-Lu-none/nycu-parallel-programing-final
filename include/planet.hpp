#ifndef PLANET_HPP
#define PLANET_HPP
#include "config.hpp"

typedef struct
{
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double r;
} Planet;

typedef struct
{
    int x[TRAIL_BUF];
    int y[TRAIL_BUF];
    int z[TRAIL_BUF];
    int head;
    int size;
} Trail;

#endif