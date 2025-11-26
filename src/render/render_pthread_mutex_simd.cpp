#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef struct {
    void* buf;
    const Camera* camera;
    const PlanetsSoA* bodies;
    const Trail* trails;
    int start_row;
    int end_row;
    int thread_id;
} RenderTaskArgs;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t cond_done;
    bool hasWork;
    bool exit;
    bool done;
    RenderTaskArgs args;
} Worker;

static Worker *workers = NULL;

void *render_thread(void *arg) {
    Worker *worker = (Worker *)arg;
    RenderTaskArgs *args = &worker->args;
    
    char threadName[32];
    snprintf(threadName, sizeof(threadName), "RenderWorker_%d", args->thread_id);
    tracy::SetThreadName(threadName);

    while (1) {
        pthread_mutex_lock(&worker->mutex);
        while (!worker->hasWork && !worker->exit) {
            pthread_cond_wait(&worker->cond, &worker->mutex);
        }
        if (worker->exit) {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }
        worker->hasWork = false;
        worker->done = false;
        pthread_mutex_unlock(&worker->mutex);

        {
            ZoneScopedN("render_worker_work_simd");
            
            const int start_row = args->start_row;
            const int end_row = args->end_row;
            color *buf = (color *)malloc(sizeof(color) * WIDTH * (end_row - start_row));
            const Camera *camera = args->camera;
            const PlanetsSoA* bodies = args->bodies;
            const Trail *trails = args->trails;

            for (int j = start_row; j < end_row; ++j) {
                for (int i = 0; i < WIDTH; ++i) {
                    vec3 pixel_center = camera->pixel00_loc + (i * camera->pixel_delta_u) + (j * camera->pixel_delta_v);
                    vec3 ray_direction = pixel_center - camera->center;
                    ray r(camera->center, ray_direction);
                    color pixel_color = get_ray_color_simd(r, *bodies, trails);
                    buf[(j - start_row) * WIDTH + i] = pixel_color;
                }
            }

            memcpy(
                (color*)args->buf + (start_row * WIDTH),
                buf,
                sizeof(color) * WIDTH * (end_row - start_row)
            );
            free(buf);
        }

        {
            ZoneScopedN("signal_completion");
            pthread_mutex_lock(&worker->mutex);
            worker->done = true;
            pthread_cond_signal(&worker->cond_done);
            pthread_mutex_unlock(&worker->mutex);
        }
    }
    return NULL;
}

void init_render_workers() {
    int t_N = config::NUM_THREADS;
    workers = (Worker *)calloc(t_N, sizeof(Worker));
    for (int i = 0; i < t_N; i++) {
        pthread_mutex_init(&workers[i].mutex, NULL);
        pthread_cond_init(&workers[i].cond, NULL);
        pthread_cond_init(&workers[i].cond_done, NULL);
        workers[i].hasWork = false;
        workers[i].exit = false;
        workers[i].args.thread_id = i;
        pthread_create(&workers[i].thread, NULL, render_thread, &workers[i]);
    }
}

void destroy_render_workers() {
    int t_N = config::NUM_THREADS;
    for (int i = 0; i < t_N; i++) {
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].exit = true;
        pthread_cond_signal(&workers[i].cond);
        pthread_mutex_unlock(&workers[i].mutex);
        pthread_join(workers[i].thread, NULL);
        pthread_mutex_destroy(&workers[i].mutex);
        pthread_cond_destroy(&workers[i].cond);
        pthread_cond_destroy(&workers[i].cond_done);
    }
    free(workers);
    workers = NULL;
}

void render(
    uint32_t *buf,
    const Camera &camera,
    const PlanetsSoA& bodies,
    const Trail* trails
)
{
    ZoneScopedN("render_pthread_mutex_simd");
    
    int t_N = config::NUM_THREADS;
    int rows_per_thread = HEIGHT / t_N;

    for (int i = 0; i < t_N; ++i) {
        workers[i].args.start_row = i * rows_per_thread;
        workers[i].args.end_row = (i == t_N - 1) ? HEIGHT : (i + 1) * rows_per_thread;
        workers[i].args.buf = buf;
        workers[i].args.camera = &camera;
        workers[i].args.bodies = &bodies;
        workers[i].args.trails = trails;
    }

    {
        ZoneScopedN("wake_up_workers");
        for (int i = 0; i < t_N; ++i) {
            pthread_mutex_lock(&workers[i].mutex);
            workers[i].hasWork = true;
            workers[i].done = false;
            pthread_cond_signal(&workers[i].cond);
            pthread_mutex_unlock(&workers[i].mutex);
        }
    }

    {
        ZoneScopedN("wait_for_workers");
        for (int i = 0; i < t_N; ++i) {
            pthread_mutex_lock(&workers[i].mutex);
            while (!workers[i].done)
                pthread_cond_wait(&workers[i].cond_done, &workers[i].mutex);
            pthread_mutex_unlock(&workers[i].mutex);
        }
    }
}
