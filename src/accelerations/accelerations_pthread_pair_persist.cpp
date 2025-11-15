#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"
#include "tracy/Tracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <pthread.h>
#include <vector>

namespace {

struct AccelerationArgs {
    const Planet *bodies = nullptr;
    std::size_t thread_id = 0;
    std::size_t thread_count = 0;
    std::size_t count = 0;
    double *ax = nullptr;
    double *ay = nullptr;
    double *az = nullptr;
};

struct Worker {
    pthread_t thread{};
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t cond_done = PTHREAD_COND_INITIALIZER;
    bool has_work = false;
    bool exit = false;
    bool done = false;
    AccelerationArgs args;
};

std::vector<Worker> workers;
std::size_t worker_count = 0;

void *worker_entry(void *arg) {
    Worker *worker = static_cast<Worker *>(arg);
    char thread_name[32];
    std::snprintf(thread_name, sizeof(thread_name), "Worker_%zu", worker->args.thread_id);
    tracy::SetThreadName(thread_name);

    while (true) {
        pthread_mutex_lock(&worker->mutex);
        while (!worker->has_work && !worker->exit) {
            pthread_cond_wait(&worker->cond, &worker->mutex);
        }
        if (worker->exit) {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }
        worker->has_work = false;
        worker->done = false;
        AccelerationArgs args = worker->args;
        pthread_mutex_unlock(&worker->mutex);

        const Planet *b = args.bodies;
        const std::size_t count = args.count;
        const std::size_t tid = args.thread_id;
        const std::size_t thread_total = args.thread_count;
        double *ax = args.ax;
        double *ay = args.ay;
        double *az = args.az;

        ZoneScopedN("compute_accelerations_persist");
        for (std::size_t i = tid; i < count; i += thread_total) {
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

        pthread_mutex_lock(&worker->mutex);
        worker->done = true;
        pthread_cond_signal(&worker->cond_done);
        pthread_mutex_unlock(&worker->mutex);
    }

    return nullptr;
}

void ensure_workers(std::size_t count) {
    std::size_t desired = std::min<std::size_t>(config::NUM_THREADS, count);
    if (worker_count == desired && !workers.empty()) {
        return;
    }

    if (!workers.empty()) {
        for (auto &worker : workers) {
            pthread_mutex_lock(&worker.mutex);
            worker.exit = true;
            pthread_cond_signal(&worker.cond);
            pthread_mutex_unlock(&worker.mutex);
            pthread_join(worker.thread, nullptr);
            pthread_mutex_destroy(&worker.mutex);
            pthread_cond_destroy(&worker.cond);
            pthread_cond_destroy(&worker.cond_done);
        }
        workers.clear();
    }

    worker_count = desired;
    if (worker_count == 0) {
        return;
    }

    workers.resize(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        Worker &worker = workers[i];
        pthread_mutex_init(&worker.mutex, nullptr);
        pthread_cond_init(&worker.cond, nullptr);
        pthread_cond_init(&worker.cond_done, nullptr);
        worker.has_work = false;
        worker.exit = false;
        worker.done = false;
        worker.args.thread_id = i;
        worker.args.thread_count = worker_count;
        pthread_create(&worker.thread, nullptr, worker_entry, &worker);
    }
}

} // namespace

void accelerations_setup(Planet *, std::size_t count) {
    ensure_workers(count);
}

void accelerations_teardown() {
    ensure_workers(0);
}

void accelerations(Planet *bodies, std::size_t count) {
    ZoneScopedN("accelerations_pthread_pair_persist");
    if (count == 0) {
        return;
    }

    ensure_workers(count);
    if (worker_count == 0) {
        return;
    }

    std::vector<std::vector<double>> partial_ax(worker_count, std::vector<double>(count, 0.0));
    std::vector<std::vector<double>> partial_ay(worker_count, std::vector<double>(count, 0.0));
    std::vector<std::vector<double>> partial_az(worker_count, std::vector<double>(count, 0.0));

    for (std::size_t i = 0; i < count; ++i) {
        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
    }

    for (std::size_t t = 0; t < worker_count; ++t) {
        Worker &worker = workers[t];
        pthread_mutex_lock(&worker.mutex);
        worker.args.bodies = bodies;
        worker.args.count = count;
        worker.args.ax = partial_ax[t].data();
        worker.args.ay = partial_ay[t].data();
        worker.args.az = partial_az[t].data();
        worker.has_work = true;
        worker.done = false;
        pthread_cond_signal(&worker.cond);
        pthread_mutex_unlock(&worker.mutex);
    }

    for (std::size_t t = 0; t < worker_count; ++t) {
        Worker &worker = workers[t];
        pthread_mutex_lock(&worker.mutex);
        while (!worker.done) {
            pthread_cond_wait(&worker.cond_done, &worker.mutex);
        }
        pthread_mutex_unlock(&worker.mutex);
    }

    for (std::size_t t = 0; t < worker_count; ++t) {
        for (std::size_t i = 0; i < count; ++i) {
            bodies[i].ax += partial_ax[t][i];
            bodies[i].ay += partial_ay[t][i];
            bodies[i].az += partial_az[t][i];
        }
    }
}

