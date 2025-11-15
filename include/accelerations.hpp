#pragma once
#include "planet.hpp"
#include "config.hpp"

void accelerations(Planet b[], int body_count);
void init_workers(int body_count);
void destroy_workers(void);
