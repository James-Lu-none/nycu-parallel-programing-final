#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp" // For G and EPSILON
#include <immintrin.h>
#include <cmath>      // For std::sqrt

void init_workers(PlanetsSoA &b) {}
void destroy_workers(PlanetsSoA &b) {}

void accelerations(PlanetsSoA &b)
{
    ZoneScopedN("accelerations");

    int n = b.count;

    // Reset accelerations
    for (int i = 0; i < n; ++i)
    {
        b.ax[i] = 0.0f;
        b.ay[i] = 0.0f;
        b.az[i] = 0.0f;
    }

    // Constants
    __m256 G_vec = _mm256_set1_ps(G);
    __m256 epsilon = _mm256_set1_ps(EPSILON);

    for (int i = 0; i < n; ++i)
    {
        __m256 xi = _mm256_set1_ps(b.x[i]);
        __m256 yi = _mm256_set1_ps(b.y[i]);
        __m256 zi = _mm256_set1_ps(b.z[i]);
        __m256 mi = _mm256_set1_ps(b.mass[i]);
        
        // Accumulators for force on body i
        __m256 axi = _mm256_setzero_ps();
        __m256 ayi = _mm256_setzero_ps();
        __m256 azi = _mm256_setzero_ps();

        int j = i + 1;
        for (; j <= n - 8; j += 8)
        {
            // Load body j properties directly from SoA
            __m256 xj = _mm256_loadu_ps(&b.x[j]);
            __m256 yj = _mm256_loadu_ps(&b.y[j]);
            __m256 zj = _mm256_loadu_ps(&b.z[j]);
            __m256 mj = _mm256_loadu_ps(&b.mass[j]);

            // Calculate distance vector
            __m256 dx = _mm256_sub_ps(xj, xi);
            __m256 dy = _mm256_sub_ps(yj, yi);
            __m256 dz = _mm256_sub_ps(zj, zi);

            // dist^2 = dx^2 + dy^2 + dz^2 + epsilon
            __m256 dist2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                _mm256_add_ps(_mm256_mul_ps(dz, dz), epsilon)
            );

            // inv_dist = 1 / sqrt(dist^2)
            __m256 inv_dist = _mm256_rsqrt_ps(dist2);
            
            // inv_dist3 = inv_dist * inv_dist * inv_dist = 1 / dist^3
            __m256 inv_dist2 = _mm256_mul_ps(inv_dist, inv_dist);
            __m256 inv_dist3 = _mm256_mul_ps(inv_dist2, inv_dist);

            // F_mag = G * mi * mj * inv_dist3
            // Force vector F = F_mag * vec(d)
            // Acceleration a = F / m
            // ai =  G * mj * inv_dist3 * vec(d)
            // aj = -G * mi * inv_dist3 * vec(d)

            __m256 factor = _mm256_mul_ps(G_vec, inv_dist3);
            __m256 factor_i = _mm256_mul_ps(factor, mj);
            __m256 factor_j = _mm256_mul_ps(factor, mi);

            __m256 fx = _mm256_mul_ps(factor_i, dx);
            __m256 fy = _mm256_mul_ps(factor_i, dy);
            __m256 fz = _mm256_mul_ps(factor_i, dz);

            axi = _mm256_add_ps(axi, fx);
            ayi = _mm256_add_ps(ayi, fy);
            azi = _mm256_add_ps(azi, fz);

            // Update body j acceleration
            // Load current acceleration
            __m256 axj = _mm256_loadu_ps(&b.ax[j]);
            __m256 ayj = _mm256_loadu_ps(&b.ay[j]);
            __m256 azj = _mm256_loadu_ps(&b.az[j]);

            // Subtract force/mass for j
            axj = _mm256_sub_ps(axj, _mm256_mul_ps(factor_j, dx));
            ayj = _mm256_sub_ps(ayj, _mm256_mul_ps(factor_j, dy));
            azj = _mm256_sub_ps(azj, _mm256_mul_ps(factor_j, dz));

            // Store back
            _mm256_storeu_ps(&b.ax[j], axj);
            _mm256_storeu_ps(&b.ay[j], ayj);
            _mm256_storeu_ps(&b.az[j], azj);
        }

        // Horizontal sum for body i
        alignas(32) float temp_ax[8];
        alignas(32) float temp_ay[8];
        alignas(32) float temp_az[8];
        _mm256_store_ps(temp_ax, axi);
        _mm256_store_ps(temp_ay, ayi);
        _mm256_store_ps(temp_az, azi);

        for (int k = 0; k < 8; ++k)
        {
            b.ax[i] += temp_ax[k];
            b.ay[i] += temp_ay[k];
            b.az[i] += temp_az[k];
        }

        // Handle remaining elements
        for (; j < n; ++j)
        {
            float dx = b.x[j] - b.x[i];
            float dy = b.y[j] - b.y[i];
            float dz = b.z[j] - b.z[i];
            float dist2 = dx*dx + dy*dy + dz*dz + EPSILON;
            float dist  = std::sqrt(dist2);

            float F = (G * b.mass[i] * b.mass[j]) / dist2;
            float fx = F * dx / dist;
            float fy = F * dy / dist;
            float fz = F * dz / dist;

            b.ax[i] += fx / b.mass[i];
            b.ay[i] += fy / b.mass[i];
            b.az[i] += fz / b.mass[i];

            b.ax[j] -= fx / b.mass[j];
            b.ay[j] -= fy / b.mass[j];
            b.az[j] -= fz / b.mass[j];
        }
    }
}
