#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

typedef struct
{
    void *buf;
    const Camera *camera;
    const vector<Planet> *bodies;
    const Trail *trails;
    int start_row;
    int end_row;
    int thread_id;
} RenderTaskArgs;

typedef struct
{
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t cond_done;
    bool hasWork;
    bool exit;
    bool done;
    RenderTaskArgs args;
} RenderWorker;

static RenderWorker *render_workers = NULL;

static void *render_thread(void *arg)
{
    RenderWorker *worker = (RenderWorker *)arg;
    RenderTaskArgs *args = &worker->args;

    char threadName[32];
    snprintf(threadName, sizeof(threadName), "RenderWorker_%d", args->thread_id);
    tracy::SetThreadName(threadName);

    while (1)
    {
        pthread_mutex_lock(&worker->mutex);

        while (!worker->hasWork && !worker->exit)
        {
            pthread_cond_wait(&worker->cond, &worker->mutex);
        }

        if (worker->exit)
        {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }

        worker->hasWork = false;
        worker->done = false;
        pthread_mutex_unlock(&worker->mutex);

        // Do the actual rendering work
        {
            ZoneScopedN("render_thread_work");

            const int start_row = args->start_row;
            const int end_row = args->end_row;
            const Camera *camera = args->camera;
            const vector<Planet> *bodies = args->bodies;
            const Trail *trails = args->trails;

            for (int j = start_row; j < end_row; ++j)
            {
                for (int i = 0; i < WIDTH; ++i)
                {
                    vec3 pixel_center = camera->pixel00_loc + (i * camera->pixel_delta_u) + (j * camera->pixel_delta_v);
                    vec3 ray_direction = pixel_center - camera->center;
                    ray r(camera->center, ray_direction);

                    color pixel_color = get_ray_color(r, *bodies, trails);
                    ((color *)args->buf)[j * WIDTH + i] = pixel_color;
                }
            }
        }

        // Signal completion
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

void init_render_workers()
{
    int t_N = config::NUM_THREADS;
    render_workers = (RenderWorker *)calloc(t_N, sizeof(RenderWorker));

    for (int i = 0; i < t_N; i++)
    {
        pthread_mutex_init(&render_workers[i].mutex, NULL);
        pthread_cond_init(&render_workers[i].cond, NULL);
        pthread_cond_init(&render_workers[i].cond_done, NULL);
        render_workers[i].hasWork = false;
        render_workers[i].exit = false;
        render_workers[i].done = false;
        render_workers[i].args.thread_id = i;
        pthread_create(&render_workers[i].thread, NULL, render_thread, &render_workers[i]);
    }
}

void destroy_render_workers()
{
    int t_N = config::NUM_THREADS;

    for (int i = 0; i < t_N; i++)
    {
        pthread_mutex_lock(&render_workers[i].mutex);
        render_workers[i].exit = true;
        pthread_cond_signal(&render_workers[i].cond);
        pthread_mutex_unlock(&render_workers[i].mutex);

        pthread_join(render_workers[i].thread, NULL);

        pthread_mutex_destroy(&render_workers[i].mutex);
        pthread_cond_destroy(&render_workers[i].cond);
        pthread_cond_destroy(&render_workers[i].cond_done);
    }

    free(render_workers);
    render_workers = NULL;
}

void render(
    uint32_t *pixels,
    const Camera &camera,
    const vector<Planet> &bodies,
    const Trail *trails
)
{
    ZoneScopedN("render_mutex");

    int t_N = config::NUM_THREADS;
    int rows_per_thread = HEIGHT / t_N;

    // Assign work to all workers
    {
        ZoneScopedN("assign_work");
        for (int i = 0; i < t_N; ++i)
        {
            render_workers[i].args.start_row = i * rows_per_thread;
            render_workers[i].args.end_row = (i == t_N - 1) ? HEIGHT : (i + 1) * rows_per_thread;
            render_workers[i].args.buf = pixels;
            render_workers[i].args.camera = &camera;
            render_workers[i].args.bodies = &bodies;
            render_workers[i].args.trails = trails;
        }
    }

    // Wake up all workers
    {
        ZoneScopedN("wake_up_workers");
        for (int t = 0; t < t_N; ++t)
        {
            pthread_mutex_lock(&render_workers[t].mutex);
            render_workers[t].hasWork = true;
            render_workers[t].done = false;
            pthread_cond_signal(&render_workers[t].cond);
            pthread_mutex_unlock(&render_workers[t].mutex);
        }
    }

    // Wait for all workers to complete
    {
        ZoneScopedN("wait_for_workers");
        for (int t = 0; t < t_N; ++t)
        {
            pthread_mutex_lock(&render_workers[t].mutex);
            while (!render_workers[t].done)
            {
                pthread_cond_wait(&render_workers[t].cond_done, &render_workers[t].mutex);
            }
            pthread_mutex_unlock(&render_workers[t].mutex);
        }
    }
}