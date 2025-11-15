#pragma once

#include <cstddef>

struct Planet;

void accelerations_setup(Planet *bodies, std::size_t count);
void accelerations(Planet *bodies, std::size_t count);
void accelerations_teardown();

