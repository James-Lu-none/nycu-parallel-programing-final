#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"
#include <immintrin.h>
#include <cstddef>

typedef struct
{
    const vector<Planet>* b;
    int t_id;
    int t_N;
    vec3 *t_acc;
} AccelerationArgs;

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    const vector<Planet>& b = *A->b;
    int n = b.size();
    
    int t_id = A->t_id;
    int t_N = A->t_N;

    vec3 *acc = A->t_acc;

    int chunk = (n + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < n) ? (i_start + chunk) : n;

    const int stride_f = sizeof(Planet) / sizeof(float);
    float* base_pos = &b[0].pos.e[0];
    float* base_mass = reinterpret_cast<float*>(
        reinterpret_cast<char*>(&b[0]) + offsetof(Planet, mass)
    );

    const __m256 G_v = _mm256_set1_ps((float)G);
    const __m256 eps_v = _mm256_set1_ps((float)EPSILON);
    const __m256 one_v = _mm256_set1_ps(1.0f);

    for (int i = i_start; i < i_end; i++)
    {
        vec3 acc_i(0.f, 0.f, 0.f);

        float mi = b[i].mass;
        float inv_mi = 1.0f / mi;

        const __m256 mi_v = _mm256_set1_ps(mi);
        const __m256 pi_x = _mm256_set1_ps(b[i].pos.x());
        const __m256 pi_y = _mm256_set1_ps(b[i].pos.y());
        const __m256 pi_z = _mm256_set1_ps(b[i].pos.z());

        int j = i+ 1;
        int vec_end = i + 1 + ((n - (i + 1)) / 8) * 8;

        for (; j < vec_end; j += 8)
        {
            __m256i vidx = _mm256_set_epi32(
                (j+7)*stride_f, (j+6)*stride_f, (j+5)*stride_f, (j+4)*stride_f,
                (j+3)*stride_f, (j+2)*stride_f, (j+1)*stride_f, (j+0)*stride_f
            );

            __m256 pj_x = _mm256_i32gather_ps(base_pos + 0, vidx, 4);
            __m256 pj_y = _mm256_i32gather_ps(base_pos + 1, vidx, 4);
            __m256 pj_z = _mm256_i32gather_ps(base_pos + 2, vidx, 4);

            __m256 dx = _mm256_sub_ps(pj_x, pi_x);
            __m256 dy = _mm256_sub_ps(pj_y, pi_y);
            __m256 dz = _mm256_sub_ps(pj_z, pi_z);

            __m256 dist2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx),
                              _mm256_mul_ps(dy, dy)),
                _mm256_mul_ps(dz, dz)
            );
            dist2 = _mm256_add_ps(dist2, eps_v);

            __m256 dist = _mm256_sqrt_ps(dist2);
            __m256 inv_dist = _mm256_div_ps(one_v, dist);

            __m256 mj = _mm256_i32gather_ps(base_mass, vidx, 4);

            __m256 tmp = _mm256_mul_ps(G_v, mi_v);
            tmp = _mm256_mul_ps(tmp, mj);
            __m256 F = _mm256_div_ps(tmp, dist2);

            __m256 scale = _mm256_mul_ps(F, inv_dist);

            __m256 fx = _mm256_mul_ps(dx, scale);
            __m256 fy = _mm256_mul_ps(dy, scale);
            __m256 fz = _mm256_mul_ps(dz, scale);

            float fx_arr[8], fy_arr[8], fz_arr[8], mj_arr[8];
            _mm256_storeu_ps(fx_arr, fx);
            _mm256_storeu_ps(fy_arr, fy);
            _mm256_storeu_ps(fz_arr, fz);
            _mm256_storeu_ps(mj_arr, mj);

            for (int k = 0; k < 8; k++)
            {
                int idx = j + k;
                vec3 force_k(fx_arr[k], fy_arr[k], fz_arr[k]);
                float inv_mj = 1.0f / mj_arr[k];

                acc_i += force_k * inv_mi;
                acc[idx] -= force_k * inv_mj;
            }
        }

        for (; j < n; j++)
        {
            vec3 dpos = b[j].pos - b[i].pos;
            float dist2 = dpos.length_squared() + EPSILON;
            float dist = std::sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;

            acc_i += force / b[i].mass;
            acc[j] -= force / b[j].mass;
        }

        acc[i] += acc_i;
    }
    return NULL;
}

void accelerations(vector<Planet> &b)
{
    ZoneScopedN("accelerations");
    int n = b.size();
    int t_N = config::NUM_THREADS > n ? n : config::NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    AccelerationArgs *args = (AccelerationArgs *)malloc(sizeof(AccelerationArgs) * t_N);

    vec3 **t_acc = new vec3*[t_N];
    for (int t = 0; t < t_N; ++t)
    {
        t_acc[t] = new vec3[n]();
    }

    for (int i = 0; i < n; ++i)
        b[i].acc = vec3(0.0, 0.0, 0.0);

    for (int t = 0; t < t_N; ++t)
    {
        args[t].b = &b;
        args[t].t_id = t;
        args[t].t_N = t_N;
        args[t].t_acc = t_acc[t];
        pthread_create(&threads[t], NULL, accelerations_thread, &args[t]);
    }

    for (int t = 0; t < t_N; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < n; ++i)
        {
            b[i].acc += t_acc[t][i];
        }
    }

    for (int t = 0; t < t_N; ++t)
    {
        delete[] t_acc[t];
    }
    delete[] t_acc;
    free(threads);
    free(args);
}
