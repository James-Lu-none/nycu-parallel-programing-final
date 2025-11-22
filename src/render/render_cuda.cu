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
CUDA_DEVICE float hit_planet_device(const Planet &p, const ray &r)
{
    vec3 oc = p.pos - r.origin();
    float a = dot(r.direction(), r.direction());
    float b = -2.0f * dot(r.direction(), oc);
    float c = dot(oc, oc) - p.r * p.r;
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

CUDA_DEVICE color get_ray_color_device(const ray &r, const Planet *bodies, const Trail *trails)
{
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        float t = hit_planet_device(bodies[i], r);
        if (t >= 0.0f)
        {
            vec3 hit_point = r.at(t);
            vec3 N = unit_vector(hit_point - bodies[i].pos);
            vec3 light_dir = unit_vector(vec3(1, 1, 0.6));
            float brightness = 0.2f + 0.8f * fmaxf(dot(N, light_dir), 0.0f);
            return {
                (uint8_t)(bodies[i].col.r * brightness),
                (uint8_t)(bodies[i].col.g * brightness),
                (uint8_t)(bodies[i].col.b * brightness),
                255};
        }
        if (hit_trail_device(trails[i], r))
        {
            return bodies[i].col;
        }
    }
    return {0, 0, 0, 255};
}

__global__ void render_kernel(color* buf, Camera cam, Planet* bodies, Trail* trails) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    // Ray generation
    // Note: Camera struct is passed by value, so we use its members directly.
    // We assume the camera is already updated on host.
    
    vec3 pixel_center = cam.pixel00_loc + ((float)x * cam.pixel_delta_u) + ((float)y * cam.pixel_delta_v);
    vec3 ray_direction = pixel_center - cam.center;
    ray r(cam.center, ray_direction);

    color pixel_color = get_ray_color_device(r, bodies, trails);
    
    // buf is 1D array of colors
    buf[y * WIDTH + x] = pixel_color;
}

static Planet* d_render_bodies = nullptr;
static Trail* d_render_trails = nullptr;
static color* d_render_buf = nullptr;

void render(
    void *buf,
    const Camera &camera,
    const vector<Planet>& bodies,
    const Trail* trails
)
{
    // Allocate device memory if needed
    if (d_render_bodies == nullptr) {
        cudaMalloc(&d_render_bodies, NUM_BODIES * sizeof(Planet));
        cudaMalloc(&d_render_trails, NUM_BODIES * sizeof(Trail));
        cudaMalloc(&d_render_buf, WIDTH * HEIGHT * sizeof(color));
    }

    // Copy data to device
    cudaMemcpy(d_render_bodies, bodies.data(), NUM_BODIES * sizeof(Planet), cudaMemcpyHostToDevice);
    if (trails != nullptr) {
        cudaMemcpy(d_render_trails, trails, NUM_BODIES * sizeof(Trail), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_render_trails, 0, NUM_BODIES * sizeof(Trail));
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    render_kernel<<<gridSize, blockSize>>>(d_render_buf, camera, d_render_bodies, d_render_trails);

    // Copy result back
    cudaMemcpy(buf, d_render_buf, WIDTH * HEIGHT * sizeof(color), cudaMemcpyDeviceToHost);
}
