#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"
#include <immintrin.h>
#include <cmath>
#include <pthread.h>

typedef struct
{
    PlanetsSoA* b;
    int t_id;
    int t_N;
    float *t_ax;
    float *t_ay;
    float *t_az;
} AccelerationArgs;

void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    PlanetsSoA& b = *A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;
    float *ax = A->t_ax;
    float *ay = A->t_ay;
    float *az = A->t_az;

    int n = b.count;

    __m256 G_vec = _mm256_set1_ps(G);
    __m256 epsilon = _mm256_set1_ps(EPSILON);

    // Interleaved: for (int i = t_id; i < n; i += t_N)
    for (int i = t_id; i < n; i += t_N)
    {
        __m256 xi = _mm256_set1_ps(b.x[i]);
        __m256 yi = _mm256_set1_ps(b.y[i]);
        __m256 zi = _mm256_set1_ps(b.z[i]);
        __m256 mi = _mm256_set1_ps(b.mass[i]);

        __m256 axi = _mm256_setzero_ps();
        __m256 ayi = _mm256_setzero_ps();
        __m256 azi = _mm256_setzero_ps();

        int j = i + 1;
        for (; j <= n - 8; j += 8)
        {
            __m256 xj = _mm256_loadu_ps(&b.x[j]);
            __m256 yj = _mm256_loadu_ps(&b.y[j]);
            __m256 zj = _mm256_loadu_ps(&b.z[j]);
            __m256 mj = _mm256_loadu_ps(&b.mass[j]);

            __m256 dx = _mm256_sub_ps(xj, xi);
            __m256 dy = _mm256_sub_ps(yj, yi);
            __m256 dz = _mm256_sub_ps(zj, zi);

            __m256 dist2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                _mm256_add_ps(_mm256_mul_ps(dz, dz), epsilon));

            __m256 inv_dist = _mm256_rsqrt_ps(dist2);
            __m256 inv_dist2 = _mm256_mul_ps(inv_dist, inv_dist);
            __m256 inv_dist3 = _mm256_mul_ps(inv_dist2, inv_dist);

            __m256 factor = _mm256_mul_ps(G_vec, inv_dist3);
            __m256 factor_i = _mm256_mul_ps(factor, mj);
            __m256 factor_j = _mm256_mul_ps(factor, mi);

            __m256 fx = _mm256_mul_ps(factor_i, dx);
            __m256 fy = _mm256_mul_ps(factor_i, dy);
            __m256 fz = _mm256_mul_ps(factor_i, dz);

            axi = _mm256_add_ps(axi, fx);
            ayi = _mm256_add_ps(ayi, fy);
            azi = _mm256_add_ps(azi, fz);

            // Update body j acceleration (thread local)
            __m256 axj = _mm256_loadu_ps(&ax[j]);
            __m256 ayj = _mm256_loadu_ps(&ay[j]);
            __m256 azj = _mm256_loadu_ps(&az[j]);

            axj = _mm256_sub_ps(axj, _mm256_mul_ps(factor_j, dx));
            ayj = _mm256_sub_ps(ayj, _mm256_mul_ps(factor_j, dy));
            azj = _mm256_sub_ps(azj, _mm256_mul_ps(factor_j, dz));

            _mm256_storeu_ps(&ax[j], axj);
            _mm256_storeu_ps(&ay[j], ayj);
            _mm256_storeu_ps(&az[j], azj);
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
            ax[i] += temp_ax[k];
            ay[i] += temp_ay[k];
            az[i] += temp_az[k];
        }

        // Handle remaining elements
        for (; j < n; ++j)
        {
            float dx = b.x[j] - b.x[i];
            float dy = b.y[j] - b.y[i];
            float dz = b.z[j] - b.z[i];
            float dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
            float dist = std::sqrt(dist2);

            float F = (G * b.mass[i] * b.mass[j]) / dist2;
            float fx = F * dx / dist;
            float fy = F * dy / dist;
            float fz = F * dz / dist;

            ax[i] += fx / b.mass[i];
            ay[i] += fy / b.mass[i];
            az[i] += fz / b.mass[i];

            ax[j] -= fx / b.mass[j];
            ay[j] -= fy / b.mass[j];
            az[j] -= fz / b.mass[j];
        }
    }
    return NULL;
}

void init_workers(PlanetsSoA &b) {}
void destroy_workers(PlanetsSoA &b) {}

void accelerations(PlanetsSoA &b)
{
    ZoneScopedN("accelerations");
    int n = b.count;
    int t_N = config::NUM_THREADS > n ? n : config::NUM_THREADS;

    pthread_t *threads = new pthread_t[t_N];
    AccelerationArgs *args = new AccelerationArgs[t_N];
    
    float **t_ax = new float*[t_N];
    float **t_ay = new float*[t_N];
    float **t_az = new float*[t_N];

    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = new float[n]();
        t_ay[t] = new float[n]();
        t_az[t] = new float[n]();

        args[t].b = &b;
        args[t].t_id = t;
        args[t].t_N = t_N;
        args[t].t_ax = t_ax[t];
        args[t].t_ay = t_ay[t];
        args[t].t_az = t_az[t];

        pthread_create(&threads[t], NULL, accelerations_thread, &args[t]);
    }

    for (int i = 0; i < n; ++i)
    {
        b.ax[i] = 0.0f;
        b.ay[i] = 0.0f;
        b.az[i] = 0.0f;
    }

    for (int t = 0; t < t_N; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < n; ++i)
        {
            b.ax[i] += t_ax[t][i];
            b.ay[i] += t_ay[t][i];
            b.az[i] += t_az[t][i];
        }
        delete[] t_ax[t];
        delete[] t_ay[t];
        delete[] t_az[t];
    }

    delete[] t_ax;
    delete[] t_ay;
    delete[] t_az;
    delete[] threads;
    delete[] args;
}
