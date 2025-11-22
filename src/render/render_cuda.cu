#ifdef __CUDACC__
#define __builtin_ia32_serialize()
#endif

#define SKIP_TRACY

#include "render.hpp"
#include "planet.hpp"
#include "camera.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <cstdio>

#ifndef ZoneScopedN
#define ZoneScopedN(x)
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

// Device helper functions
CUDA_DEVICE float hit_planet_device(
    vec3 origin, vec3 direction,
    float px, float py, float pz, float pr)
{
    vec3 oc = vec3(px, py, pz) - origin;
    float a = dot(direction, direction);
    float b = -2.0f * dot(direction, oc);
    float c = dot(oc, oc) - pr * pr;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
    {
        return -1.0f;
    }
    else
    {
        return (-b - sqrtf(discriminant)) / (2.0f * a);
    }
}

CUDA_DEVICE bool hit_trail_device(const Trail &t, const ray &r)
{
    float radius_squared = 4.0f; // radius 2
    for (int i = 0; i < t.size; ++i)
    {
        vec3 oc = t.pos[i] - r.origin();
        float a = dot(r.direction(), r.direction());
        float b = -2.0f * dot(r.direction(), oc);
        float c = dot(oc, oc) - radius_squared;
        float discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            return true;
        }
    }
    return false;
}

CUDA_DEVICE color get_ray_color_device(
    const ray &r,
    int n,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ radius,
    const uint8_t* __restrict__ col_r,
    const uint8_t* __restrict__ col_g,
    const uint8_t* __restrict__ col_b,
    const Trail *trails)
{
    for (int i = 0; i < n; ++i)
    {
        float t = hit_planet_device(r.origin(), r.direction(), x[i], y[i], z[i], radius[i]);
        if (t >= 0.0f)
        {
            vec3 hit_point = r.at(t);
            vec3 N = unit_vector(hit_point - vec3(x[i], y[i], z[i]));
            vec3 light_dir = unit_vector(vec3(1, 1, 0.6));
            float brightness = 0.2f + 0.8f * fmaxf(dot(N, light_dir), 0.0f);
            return {
                (uint8_t)(col_r[i] * brightness),
                (uint8_t)(col_g[i] * brightness),
                (uint8_t)(col_b[i] * brightness),
                255};
        }
        if (trails && hit_trail_device(trails[i], r))
        {
            return {col_r[i], col_g[i], col_b[i], 255};
        }
    }
    return {0, 0, 0, 255};
}

__global__ void render_kernel(
    color* buf, Camera cam,
    int n,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ radius,
    const uint8_t* __restrict__ col_r,
    const uint8_t* __restrict__ col_g,
    const uint8_t* __restrict__ col_b,
    Trail* trails)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= WIDTH || j >= HEIGHT) return;

    // Ray generation
    vec3 pixel_center = cam.pixel00_loc + ((float)i * cam.pixel_delta_u) + ((float)j * cam.pixel_delta_v);
    vec3 ray_direction = pixel_center - cam.center;
    ray r(cam.center, ray_direction);

    color pixel_color = get_ray_color_device(r, n, x, y, z, radius, col_r, col_g, col_b, trails);
    
    buf[j * WIDTH + i] = pixel_color;
}

static float *d_x = nullptr;
static float *d_y = nullptr;
static float *d_z = nullptr;
static float *d_r = nullptr;
static uint8_t *d_col_r = nullptr;
static uint8_t *d_col_g = nullptr;
static uint8_t *d_col_b = nullptr;

static Trail* d_render_trails = nullptr;
static color* d_render_buf = nullptr;
static int d_capacity = 0;

void render(
    uint32_t *buf,
    const Camera &camera,
    const PlanetsSoA& bodies,
    const Trail* trails
)
{
    int n = bodies.count;

    // Allocate device memory if needed
    if (d_capacity < n) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_z) cudaFree(d_z);
        if (d_r) cudaFree(d_r);
        if (d_col_r) cudaFree(d_col_r);
        if (d_col_g) cudaFree(d_col_g);
        if (d_col_b) cudaFree(d_col_b);
        if (d_render_trails) cudaFree(d_render_trails);
        if (d_render_buf) cudaFree(d_render_buf);

        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        cudaMalloc(&d_z, n * sizeof(float));
        cudaMalloc(&d_r, n * sizeof(float));
        cudaMalloc(&d_col_r, n * sizeof(uint8_t));
        cudaMalloc(&d_col_g, n * sizeof(uint8_t));
        cudaMalloc(&d_col_b, n * sizeof(uint8_t));
        cudaMalloc(&d_render_trails, n * sizeof(Trail));
        cudaMalloc(&d_render_buf, WIDTH * HEIGHT * sizeof(color));
        
        d_capacity = n;
    }

    // Copy data to device
    cudaMemcpy(d_x, bodies.x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, bodies.y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, bodies.z, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, bodies.r, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_r, bodies.col_r, n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_g, bodies.col_g, n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_b, bodies.col_b, n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (trails != nullptr) {
        cudaMemcpy(d_render_trails, trails, n * sizeof(Trail), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_render_trails, 0, n * sizeof(Trail));
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    render_kernel<<<gridSize, blockSize>>>(
        d_render_buf, camera, n,
        d_x, d_y, d_z, d_r,
        d_col_r, d_col_g, d_col_b,
        d_render_trails
    );

    // Copy result back
    cudaMemcpy(buf, d_render_buf, WIDTH * HEIGHT * sizeof(color), cudaMemcpyDeviceToHost);
}