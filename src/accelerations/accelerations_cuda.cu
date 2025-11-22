#ifdef __CUDACC__
#define __builtin_ia32_serialize()
#endif

#define SKIP_TRACY

#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#ifndef ZoneScopedN
#define ZoneScopedN(x)
#endif

#define BLOCK_SIZE 1024

__global__ void accelerations_kernel(Planet* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    vec3 acc(0.0f, 0.0f, 0.0f);
    vec3 pos_i;
    float mass_i;

    if (i < n) {
        pos_i = b[i].pos;
        mass_i = b[i].mass;
    }

    __shared__ Planet cache[BLOCK_SIZE];

    // Loop over all tiles
    for (int tile = 0; tile < gridDim.x; ++tile) {
        // Load tile into shared memory
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            cache[threadIdx.x] = b[idx];
        } else {
            // Zero out for safety
            cache[threadIdx.x].mass = 0.0f;
            cache[threadIdx.x].pos = vec3(0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int j_idx = tile * blockDim.x + j;
                if (j_idx >= n) break;

                if (i == j_idx) continue;

                vec3 dpos = cache[j].pos - pos_i;
                float dist2 = dpos.length_squared() + EPSILON;
                float inv_dist = rsqrtf(dist2);
                float dist = dist2 * inv_dist;

                float F = (G * mass_i * cache[j].mass) / dist2;
                vec3 force = F * dpos / dist;
                acc += force / mass_i;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        b[i].acc = acc;
    }
}

__global__ void integrate_kernel(Planet* b, int n, float dt, bool first_half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (first_half) {
        b[i].vel += 0.5f * b[i].acc * dt;
        b[i].pos += b[i].vel * dt;
    } else {
        b[i].vel += 0.5f * b[i].acc * dt;
    }
}

static Planet* d_planets = nullptr;
static int d_capacity = 0;

void accelerations(vector<Planet> &b) {
    ZoneScopedN("accelerations");
    int n = b.size();

    // If d_planets is already allocated (by accelerations_integrate), use it.
    // Otherwise, allocate and copy (fallback for non-integrated usage).
    bool managed = (d_planets != nullptr && d_capacity >= n);

    if (!managed) {
        if (d_planets) cudaFree(d_planets);
        cudaError_t err = cudaMalloc(&d_planets, n * sizeof(Planet));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        d_capacity = n;
        
        err = cudaMemcpy(d_planets, b.data(), n * sizeof(Planet), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_planets, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Always copy back for now
    err = cudaMemcpy(b.data(), d_planets, n * sizeof(Planet), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    if (!managed) {
        cudaFree(d_planets);
        d_planets = nullptr;
        d_capacity = 0;
    }
}

void accelerations_integrate(Planet* b, int n, float dt) {
    static bool initialized = false;
    
    if (!initialized || d_capacity < n) {
        if (d_planets) cudaFree(d_planets);
        cudaError_t err = cudaMalloc(&d_planets, n * sizeof(Planet));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        d_capacity = n;

        // Initial copy from host to device
        err = cudaMemcpy(d_planets, b, n * sizeof(Planet), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        initialized = true;
    }

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 1. First half kick + drift
    integrate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_planets, n, dt, true);
    cudaDeviceSynchronize();

    // 2. Calculate accelerations
    accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_planets, n);
    cudaDeviceSynchronize();

    // 3. Second half kick
    integrate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_planets, n, dt, false);
    cudaDeviceSynchronize();

    // Copy back to host for rendering
    cudaError_t err = cudaMemcpy(b, d_planets, n * sizeof(Planet), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
