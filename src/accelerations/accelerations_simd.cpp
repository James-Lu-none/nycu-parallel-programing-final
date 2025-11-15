#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"
#include "tracy/Tracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <vector>

namespace {

inline double horizontal_sum(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    vlow = _mm_add_sd(vlow, high64);
    return _mm_cvtsd_f64(vlow);
}

struct alignas(32) PackedValues {
    double x[4];
    double y[4];
    double z[4];
    double mass[4];
};

} // namespace

void accelerations_setup(Planet *, std::size_t) {}

void accelerations_teardown() {}

void accelerations(Planet *bodies, std::size_t count) {
    ZoneScopedN("accelerations_simd");
    if (count == 0) {
        return;
    }

    for (std::size_t i = 0; i < count; ++i) {
        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
    }

    const __m256d epsilon = _mm256_set1_pd(config::EPSILON);
    const __m256d g_const = _mm256_set1_pd(config::G);

    for (std::size_t i = 0; i < count; ++i) {
        __m256d xi = _mm256_set1_pd(bodies[i].x);
        __m256d yi = _mm256_set1_pd(bodies[i].y);
        __m256d zi = _mm256_set1_pd(bodies[i].z);

        __m256d accx = _mm256_setzero_pd();
        __m256d accy = _mm256_setzero_pd();
        __m256d accz = _mm256_setzero_pd();

        double scalar_ax = 0.0;
        double scalar_ay = 0.0;
        double scalar_az = 0.0;

        std::size_t j = 0;
        while (j + 3 < count) {
            if (i >= j && i <= j + 3) {
                for (std::size_t k = j; k < j + 4 && k < count; ++k) {
                    if (k == i) {
                        continue;
                    }
                    double dx = bodies[k].x - bodies[i].x;
                    double dy = bodies[k].y - bodies[i].y;
                    double dz = bodies[k].z - bodies[i].z;
                    double dist2 = dx * dx + dy * dy + dz * dz + config::EPSILON;
                    double dist = std::sqrt(dist2);
                    double scale = (config::G * bodies[k].mass) / (dist2 * dist);
                    scalar_ax += scale * dx;
                    scalar_ay += scale * dy;
                    scalar_az += scale * dz;
                }
                j += 4;
                continue;
            }

            PackedValues pack{};
            for (int lane = 0; lane < 4; ++lane) {
                pack.x[lane] = bodies[j + lane].x;
                pack.y[lane] = bodies[j + lane].y;
                pack.z[lane] = bodies[j + lane].z;
                pack.mass[lane] = bodies[j + lane].mass;
            }

            __m256d xj = _mm256_load_pd(pack.x);
            __m256d yj = _mm256_load_pd(pack.y);
            __m256d zj = _mm256_load_pd(pack.z);
            __m256d mj = _mm256_load_pd(pack.mass);

            __m256d dx = _mm256_sub_pd(xj, xi);
            __m256d dy = _mm256_sub_pd(yj, yi);
            __m256d dz = _mm256_sub_pd(zj, zi);

            __m256d dist2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                                          _mm256_add_pd(_mm256_mul_pd(dz, dz), epsilon));
            __m256d dist = _mm256_sqrt_pd(dist2);
            __m256d denom = _mm256_mul_pd(dist2, dist);
            __m256d scale = _mm256_div_pd(_mm256_mul_pd(g_const, mj), denom);

            accx = _mm256_add_pd(accx, _mm256_mul_pd(dx, scale));
            accy = _mm256_add_pd(accy, _mm256_mul_pd(dy, scale));
            accz = _mm256_add_pd(accz, _mm256_mul_pd(dz, scale));

            j += 4;
        }

        for (; j < count; ++j) {
            if (j == i) {
                continue;
            }
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + config::EPSILON;
            double dist = std::sqrt(dist2);
            double scale = (config::G * bodies[j].mass) / (dist2 * dist);
            scalar_ax += scale * dx;
            scalar_ay += scale * dy;
            scalar_az += scale * dz;
        }

        bodies[i].ax += horizontal_sum(accx) + scalar_ax;
        bodies[i].ay += horizontal_sum(accy) + scalar_ay;
        bodies[i].az += horizontal_sum(accz) + scalar_az;
    }
}

