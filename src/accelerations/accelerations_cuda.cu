#ifdef __CUDACC__
#define __builtin_ia32_serialize()
#endif

#define SKIP_TRACY

#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <cstdio>

#ifndef ZoneScopedN
#define ZoneScopedN(x)
#endif

#define BLOCK_SIZE 128

__global__ void accelerations_kernel(
    int n,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float my_x, my_y, my_z, my_mass;
    float my_ax = 0.0f;
    float my_ay = 0.0f;
    float my_az = 0.0f;

    if (i < n) {
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_mass = mass[i];
    }

    __shared__ float sh_x[BLOCK_SIZE];
    __shared__ float sh_y[BLOCK_SIZE];
    __shared__ float sh_z[BLOCK_SIZE];
    __shared__ float sh_mass[BLOCK_SIZE];

    for (int tile = 0; tile < gridDim.x; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            sh_x[threadIdx.x] = x[idx];
            sh_y[threadIdx.x] = y[idx];
            sh_z[threadIdx.x] = z[idx];
            sh_mass[threadIdx.x] = mass[idx];
        } else {
            sh_mass[threadIdx.x] = 0.0f; // Padding
             // positions don't matter if mass is 0, but good to init
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_z[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int j_idx = tile * blockDim.x + j;
                // if (j_idx >= n) break; // Handled by mass=0 check implicitly or explicit check?
                // Explicit check is safer if n is not multiple of BLOCK_SIZE
                if (j_idx >= n) break;

                if (i == j_idx) continue;

                float dx = sh_x[j] - my_x;
                float dy = sh_y[j] - my_y;
                float dz = sh_z[j] - my_z;
                float dist2 = dx*dx + dy*dy + dz*dz + EPSILON;
                float inv_dist = rsqrtf(dist2);
                float dist = dist2 * inv_dist; // dist = sqrt(dist2)

                float F = (G * my_mass * sh_mass[j]) / dist2;
                float scale = F / dist; // F * (dpos / dist) = (F/dist) * dpos
                
                my_ax += scale * dx;
                my_ay += scale * dy;
                my_az += scale * dz;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        // F = ma => a = F/m
        float inv_m = 1.0f / my_mass;
        ax[i] = my_ax * inv_m;
        ay[i] = my_ay * inv_m;
        az[i] = my_az * inv_m;
    }
}

// Device memory pointers
static float *d_x = nullptr;
static float *d_y = nullptr;
static float *d_z = nullptr;
static float *d_mass = nullptr;
static float *d_ax = nullptr;
static float *d_ay = nullptr;
static float *d_az = nullptr;
static int d_capacity = 0;

void init_workers(PlanetsSoA &b) {}
void destroy_workers(PlanetsSoA &b) {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_mass) cudaFree(d_mass);
    if (d_ax) cudaFree(d_ax);
    if (d_ay) cudaFree(d_ay);
    if (d_az) cudaFree(d_az);
    d_x = nullptr;
    d_capacity = 0;
}

void accelerations(PlanetsSoA &b) {
    ZoneScopedN("accelerations");
    int n = b.count;

    if (d_capacity < n) {
        destroy_workers(b);
        
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        cudaMalloc(&d_z, n * sizeof(float));
        cudaMalloc(&d_mass, n * sizeof(float));
        cudaMalloc(&d_ax, n * sizeof(float));
        cudaMalloc(&d_ay, n * sizeof(float));
        cudaMalloc(&d_az, n * sizeof(float));
        
        d_capacity = n;
    }

    // Copy positions and mass to device
    cudaMemcpy(d_x, b.x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, b.y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, b.z, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, b.mass, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        n, d_x, d_y, d_z, d_mass, d_ax, d_ay, d_az
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy accelerations back to host
    cudaMemcpy(b.ax, d_ax, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.ay, d_ay, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.az, d_az, n * sizeof(float), cudaMemcpyDeviceToHost);
}