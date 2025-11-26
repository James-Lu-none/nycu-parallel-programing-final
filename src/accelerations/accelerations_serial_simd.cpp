#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"

#include <immintrin.h>  // AVX
#include <cstddef>      // offsetof

void accelerations(vector<Planet> &b)
{
    ZoneScopedN("accelerations");
    int n = b.size();

    // 1. clear acc
    for (int i = 0; i < n; ++i)
        b[i].acc = vec3(0.f, 0.f, 0.f);

    // stride of Planet in float count
    const int stride_f = sizeof(Planet) / sizeof(float);

    float* base_pos  = &b[0].pos.e[0];
    float* base_mass = reinterpret_cast<float*>(
        reinterpret_cast<char*>(&b[0]) + offsetof(Planet, mass)
    );

    const __m256 G_v   = _mm256_set1_ps((float)G);
    const __m256 eps_v = _mm256_set1_ps((float)EPSILON);
    const __m256 one_v = _mm256_set1_ps(1.0f);

    for (int i = 0; i < n; i++)
    {
        vec3 acc_i(0.f, 0.f, 0.f);

        float mi     = b[i].mass;
        float inv_mi = 1.0f / mi;

        const __m256 mi_v = _mm256_set1_ps(mi);
        const __m256 pi_x = _mm256_set1_ps(b[i].pos.x());
        const __m256 pi_y = _mm256_set1_ps(b[i].pos.y());
        const __m256 pi_z = _mm256_set1_ps(b[i].pos.z());

        int j = i + 1;

        int vec_end = i + 1 + ((n - (i + 1)) / 8) * 8;

        for (; j < vec_end; j += 8)
        {
            __m256i vidx = _mm256_set_epi32(
                (j+7)*stride_f, (j+6)*stride_f, (j+5)*stride_f, (j+4)*stride_f,
                (j+3)*stride_f, (j+2)*stride_f, (j+1)*stride_f, (j+0)*stride_f
            );

            // gather positions
            __m256 pj_x = _mm256_i32gather_ps(base_pos + 0, vidx, 4);
            __m256 pj_y = _mm256_i32gather_ps(base_pos + 1, vidx, 4);
            __m256 pj_z = _mm256_i32gather_ps(base_pos + 2, vidx, 4);

            // dpos = pj - pi
            __m256 dx = _mm256_sub_ps(pj_x, pi_x);
            __m256 dy = _mm256_sub_ps(pj_y, pi_y);
            __m256 dz = _mm256_sub_ps(pj_z, pi_z);

            // dist2 = |dpos|^2 + eps
            __m256 dist2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx),
                              _mm256_mul_ps(dy, dy)),
                _mm256_mul_ps(dz, dz)
            );
            dist2 = _mm256_add_ps(dist2, eps_v);

            // dist = sqrt(dist2)
            __m256 dist = _mm256_sqrt_ps(dist2);
            __m256 inv_dist = _mm256_div_ps(one_v, dist);

            // mj
            __m256 mj = _mm256_i32gather_ps(base_mass, vidx, 4);

            // F = G * mi * mj / dist2
            __m256 tmp = _mm256_mul_ps(G_v, mi_v);
            tmp        = _mm256_mul_ps(tmp, mj);
            __m256 F   = _mm256_div_ps(tmp, dist2);

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
                b[idx].acc -= force_k * inv_mj;
            }
        }

        // scalar tail
        for (; j < n; j++)
        {
            vec3 dpos = b[j].pos - b[i].pos;
            float dist2 = dpos.length_squared() + EPSILON;
            float dist  = std::sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;

            acc_i += force / b[i].mass;
            b[j].acc -= force / b[j].mass;
        }

        b[i].acc += acc_i;
    }
}
