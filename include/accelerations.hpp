#pragma once
#include "planet.hpp"
#include "config.hpp"

void accelerations(vector<Planet>& b);
void init_workers(vector<Planet> &b);
void destroy_workers(vector<Planet> &b);