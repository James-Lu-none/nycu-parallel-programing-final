#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp" // For G and EPSILON
#include <immintrin.h>
#include <cmath>      // For std::sqrt

void init_workers(PlanetsSoA &b) {}
void destroy_workers(PlanetsSoA &b) {}

using accelerations_func = void(*)(PlanetsSoA &);
static accelerations_func accelerations_impl = accelerations_simd;

void accelerations(PlanetsSoA &b)
{
    accelerations_impl(b);
}