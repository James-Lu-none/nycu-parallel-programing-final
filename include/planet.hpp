#ifndef PLANET_HPP
#define PLANET_HPP

typedef struct
{
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double r;
} Planet;

#endif