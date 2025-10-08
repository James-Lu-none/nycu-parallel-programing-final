#pragma once
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <SDL2/SDL.h>
#include <pthread.h>
#include <ctime>
#include <cstdlib>
#include "planet.hpp"
#include "accelerations.hpp"
#include "tracy/Tracy.hpp"

#define WIDTH 1200
#define HEIGHT 800

#define NUM_BODIES 3
#define TRAIL_BUF 100
#define MIN_DIST 1.5

#define G 10000.0
#define EPSILON 1e-6

#define COL_BLACK 0x00000000

#define NUM_THREADS 4

#define view_z 100.0
