#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"

#include <immintrin.h>  // AVX2
#include <cstddef>      // offsetof

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations");

    // 1. 先把所有加速度清 0
    for (int i = 0; i < NUM_BODIES; ++i)
        b[i].acc = vec3(0.0, 0.0, 0.0);

    // 2. 準備一些常數 / 指標給 gather 用
    const long long stride_d = static_cast<long long>(sizeof(Planet) / sizeof(double));

    // Planet 陣列視為 double 陣列的起點
    double *base_pos  = &b[0].pos.e[0]; // pos.x 的位址
    // mass 欄位的 double offset
    const std::size_t mass_off_bytes = offsetof(Planet, mass);
    double *base_mass = reinterpret_cast<double*>(
        reinterpret_cast<char*>(&b[0]) + mass_off_bytes
    );

    const __m256d G_v   = _mm256_set1_pd(G);
    const __m256d eps_v = _mm256_set1_pd(EPSILON);
    const __m256d one_v = _mm256_set1_pd(1.0);

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        // 對 body i 的加速度用一個暫存 vec3 來累加
        vec3 acc_i(0.0, 0.0, 0.0);

        const double mi     = b[i].mass;
        const double inv_mi = 1.0 / mi;

        const __m256d mi_v = _mm256_set1_pd(mi);
        const __m256d pi_x = _mm256_set1_pd(b[i].pos.x());
        const __m256d pi_y = _mm256_set1_pd(b[i].pos.y());
        const __m256d pi_z = _mm256_set1_pd(b[i].pos.z());

        int j = i + 1;

        // 一次處理 4 個 j： [j, j+1, j+2, j+3]
        const int vec_end = i + 1 + ((NUM_BODIES - (i + 1)) / 4) * 4;

        for (; j < vec_end; j += 4)
        {
            // -------- gather positions of j..j+3 --------
            long long idx0 = (long long)(j + 0) * stride_d;
            long long idx1 = (long long)(j + 1) * stride_d;
            long long idx2 = (long long)(j + 2) * stride_d;
            long long idx3 = (long long)(j + 3) * stride_d;

            __m256i vidx = _mm256_set_epi64x(idx3, idx2, idx1, idx0);

            __m256d pj_x = _mm256_i64gather_pd(base_pos + 0, vidx, 8);
            __m256d pj_y = _mm256_i64gather_pd(base_pos + 1, vidx, 8);
            __m256d pj_z = _mm256_i64gather_pd(base_pos + 2, vidx, 8);

            // dpos = pj - pi
            __m256d dx = _mm256_sub_pd(pj_x, pi_x);
            __m256d dy = _mm256_sub_pd(pj_y, pi_y);
            __m256d dz = _mm256_sub_pd(pj_z, pi_z);

            // dist2 = |dpos|^2 + EPSILON
            __m256d dx2   = _mm256_mul_pd(dx, dx);
            __m256d dy2   = _mm256_mul_pd(dy, dy);
            __m256d dz2   = _mm256_mul_pd(dz, dz);
            __m256d dist2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
            dist2         = _mm256_add_pd(dist2, eps_v);

            // dist = sqrt(dist2)
            __m256d dist     = _mm256_sqrt_pd(dist);
            dist              = _mm256_sqrt_pd(dist2);
            __m256d inv_dist = _mm256_div_pd(one_v, dist); // 1 / dist

            // masses of j..j+3
            __m256d mj = _mm256_i64gather_pd(base_mass, vidx, 8);

            // F = (G * mi * mj) / dist2
            __m256d tmp = _mm256_mul_pd(G_v, mi_v);
            tmp         = _mm256_mul_pd(tmp, mj);
            __m256d F   = _mm256_div_pd(tmp, dist2);

            // scale = F / dist
            __m256d scale = _mm256_mul_pd(F, inv_dist);

            // force components on each lane
            __m256d fx = _mm256_mul_pd(dx, scale);
            __m256d fy = _mm256_mul_pd(dy, scale);
            __m256d fz = _mm256_mul_pd(dz, scale);

            // 把 SIMD 結果存回 scalar 陣列，方便更新 Planet 結構
            double fx_arr[4], fy_arr[4], fz_arr[4], mj_arr[4];
            _mm256_storeu_pd(fx_arr, fx);
            _mm256_storeu_pd(fy_arr, fy);
            _mm256_storeu_pd(fz_arr, fz);
            _mm256_storeu_pd(mj_arr, mj);

            // 逐一更新 j..j+3 的加速度，並累加到 i 的 acc_i
            for (int k = 0; k < 4; ++k)
            {
                const int idx = j + k;

                vec3 force_k(fx_arr[k], fy_arr[k], fz_arr[k]);

                const double inv_mj = 1.0 / mj_arr[k];

                // b[i].acc += force / mi;
                acc_i += force_k * inv_mi;

                // b[j].acc -= force / mj;
                b[idx].acc -= force_k * inv_mj;
            }
        }

        // scalar tail：剩下沒湊到 4 的 j
        for (; j < NUM_BODIES; ++j)
        {
            vec3 dpos  = b[j].pos - b[i].pos;
            double dist2 = dpos.length_squared() + EPSILON;
            double dist  = std::sqrt(dist2);

            double F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;

            acc_i      += force / b[i].mass;
            b[j].acc   -= force / b[j].mass;
        }

        // 把累積的 acc_i 寫回 b[i].acc
        b[i].acc += acc_i;
    }
}
