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
    int count;

    alignas(32) float *x;
    alignas(32) float *y;
    alignas(32) float *z;

    alignas(32) float *vx;
    alignas(32) float *vy;
    alignas(32) float *vz;

    alignas(32) float *ax;
    alignas(32) float *ay;
    alignas(32) float *az;

    alignas(32) float *mass;
    alignas(32) float *r;

    // store color as 0–255 bytes
    uint8_t *col_r;
    uint8_t *col_g;
    uint8_t *col_b;
    uint8_t *col_a;
} PlanetsSoA;

typedef struct
{
    point3 pos[TRAIL_BUF];
    int head;
    int size;
} Trail;

void load_planets_from_file(const char *filename, vector<Planet>& b);
void load_planets_to_SoA(vector<Planet> &bodies, PlanetsSoA &bodies_soa);
vec3 get_center_of_mass(vector<Planet>& b);
void trail_push(Trail *t, vec3 pos);

#endif