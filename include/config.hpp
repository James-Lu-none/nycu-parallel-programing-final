#pragma once
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <SDL2/SDL.h>
#include <pthread.h>
#include <ctime>
#include <cstdlib>
#include <vector>
#ifndef __CUDACC__
#include "tracy/Tracy.hpp"
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

using namespace std;

namespace config
{
    extern int NUM_THREADS;
}

#define WIDTH 1200
#define HEIGHT 800

// generate NUM_BODIES bodies if no file is provided
#define NUM_BODIES 100
#define TRAIL_BUF 5
#define MIN_DIST 1.5

#define G 10000.0
#define EPSILON 1e-6

#define COL_BLACK 0x00000000

#define view_z 100.0
