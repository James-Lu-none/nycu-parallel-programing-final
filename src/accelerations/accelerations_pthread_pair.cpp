#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"
#include "tracy/Tracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <pthread.h>
#include <vector>

namespace {

struct AccelerationArgs {
    const Planet *bodies;
    std::size_t begin;
    std::size_t end;
    std::size_t count;
    double *ax;
    double *ay;
    double *az;
};

void *accelerations_thread(void *arg) {
    AccelerationArgs *A = static_cast<AccelerationArgs *>(arg);
    const Planet *b = A->bodies;
    const std::size_t begin = A->begin;
    const std::size_t end = A->end;
    const std::size_t count = A->count;
    double *ax = A->ax;
    double *ay = A->ay;
    double *az = A->az;

    for (std::size_t i = begin; i < end; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            double dx = b[j].x - b[i].x;
            double dy = b[j].y - b[i].y;
            double dz = b[j].z - b[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + config::EPSILON;
            double dist = std::sqrt(dist2);

            double F = (config::G * b[i].mass * b[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;

            ax[i] += fx / b[i].mass;
            ay[i] += fy / b[i].mass;
            az[i] += fz / b[i].mass;
            ax[j] -= fx / b[j].mass;
            ay[j] -= fy / b[j].mass;
            az[j] -= fz / b[j].mass;
        }
    }

    return nullptr;
}

} // namespace

void accelerations_setup(Planet *, std::size_t) {}

void accelerations_teardown() {}

void accelerations(Planet *bodies, std::size_t count) {
    ZoneScopedN("accelerations_pthread_pair");
    if (count == 0) {
        return;
    }

    const std::size_t thread_count = std::min<std::size_t>(config::NUM_THREADS, count);
    std::vector<pthread_t> threads(thread_count);
    std::vector<AccelerationArgs> args(thread_count);
    std::vector<std::vector<double>> partial_ax(thread_count, std::vector<double>(count, 0.0));
    std::vector<std::vector<double>> partial_ay(thread_count, std::vector<double>(count, 0.0));
    std::vector<std::vector<double>> partial_az(thread_count, std::vector<double>(count, 0.0));

    for (std::size_t i = 0; i < count; ++i) {
        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
    }

    std::size_t chunk = (count + thread_count - 1) / thread_count;
    for (std::size_t t = 0; t < thread_count; ++t) {
        std::size_t begin = t * chunk;
        std::size_t end = std::min(count, begin + chunk);
        args[t] = AccelerationArgs{bodies, begin, end, count,
                                   partial_ax[t].data(), partial_ay[t].data(), partial_az[t].data()};
        pthread_create(&threads[t], nullptr, accelerations_thread, &args[t]);
    }

    for (std::size_t t = 0; t < thread_count; ++t) {
        pthread_join(threads[t], nullptr);
    }

    for (std::size_t t = 0; t < thread_count; ++t) {
        for (std::size_t i = 0; i < count; ++i) {
            bodies[i].ax += partial_ax[t][i];
            bodies[i].ay += partial_ay[t][i];
            bodies[i].az += partial_az[t][i];
        }
    }
}

