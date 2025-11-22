#pragma once
#include "planet.hpp"
#include "config.hpp"

void accelerations(PlanetsSoA& b);
void init_workers(PlanetsSoA &b);
void destroy_workers(PlanetsSoA &b);