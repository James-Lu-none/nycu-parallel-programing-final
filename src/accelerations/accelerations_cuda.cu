#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"
#include "tracy/Tracy.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace {

Planet *d_bodies = nullptr;
std::size_t d_capacity = 0;

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

void ensure_capacity(std::size_t count) {
    if (count <= d_capacity) {
        return;
    }
    if (d_bodies != nullptr) {
        check_cuda(cudaFree(d_bodies), "cudaFree failed");
        d_bodies = nullptr;
        d_capacity = 0;
    }
    if (count == 0) {
        return;
    }
    check_cuda(cudaMalloc(&d_bodies, count * sizeof(Planet)), "cudaMalloc failed");
    d_capacity = count;
}

__global__ void compute_accelerations(Planet *bodies, std::size_t count) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    Planet bi = bodies[i];
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (std::size_t j = 0; j < count; ++j) {
        if (i == j) {
            continue;
        }
        Planet bj = bodies[j];
        double dx = bj.x - bi.x;
        double dy = bj.y - bi.y;
        double dz = bj.z - bi.z;
        double dist2 = dx * dx + dy * dy + dz * dz + config::EPSILON;
        double dist = sqrt(dist2);
        double scale = (config::G * bj.mass) / (dist2 * dist);
        ax += scale * dx;
        ay += scale * dy;
        az += scale * dz;
    }

    bodies[i].ax = ax;
    bodies[i].ay = ay;
    bodies[i].az = az;
}

} // namespace

void accelerations_setup(Planet *bodies, std::size_t count) {
    ensure_capacity(count);
    if (bodies != nullptr && count > 0) {
        check_cuda(cudaMemcpy(d_bodies, bodies, count * sizeof(Planet), cudaMemcpyHostToDevice),
                   "cudaMemcpy host->device failed");
    }
}

void accelerations_teardown() {
    if (d_bodies != nullptr) {
        check_cuda(cudaFree(d_bodies), "cudaFree failed");
        d_bodies = nullptr;
        d_capacity = 0;
    }
}

void accelerations(Planet *bodies, std::size_t count) {
    ZoneScopedN("accelerations_cuda");
    if (count == 0) {
        return;
    }

    ensure_capacity(count);

    check_cuda(cudaMemcpy(d_bodies, bodies, count * sizeof(Planet), cudaMemcpyHostToDevice),
               "cudaMemcpy host->device failed");

    int threads_per_block = 128;
    int blocks = static_cast<int>((count + threads_per_block - 1) / threads_per_block);
    compute_accelerations<<<blocks, threads_per_block>>>(d_bodies, count);
    check_cuda(cudaGetLastError(), "Kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

    check_cuda(cudaMemcpy(bodies, d_bodies, count * sizeof(Planet), cudaMemcpyDeviceToHost),
               "cudaMemcpy device->host failed");
}

