#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

typedef struct
{
    PlanetsSoA *b;
    int t_id;
    int t_N;
    float *t_ax;
    float *t_ay;
    float *t_az;
} AccelerationArgs;

typedef struct
{
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t cond_done;
    bool hasWork;
    bool exit;
    bool done;
    AccelerationArgs args;
} Worker;

static Worker *workers = NULL;

static void *accelerations_thread(void *arg)
{
    Worker *worker = (Worker *)arg;
    AccelerationArgs *A = &worker->args;
    char threadName[32];
    snprintf(threadName, sizeof(threadName), "Worker_%d", A->t_id);
    tracy::SetThreadName(threadName);
    // printf("Thread %d initialized\n", A->t_id);
    while (1)
    {
        pthread_mutex_lock(&worker->mutex);
        // printf("Thread %d entered critical session\n", A->t_id);
        while (!worker->hasWork && !worker->exit)
        {
            // printf("Thread %d has no work and waiting to be signaled\n", A->t_id);
            pthread_cond_wait(&worker->cond, &worker->mutex);
        }
        if (worker->exit)
        {
            // printf("Thread %d exiting\n", A->t_id);
            pthread_mutex_unlock(&worker->mutex);
            break;
        }
        // printf("Thread %d resetting hasWork flag and starting work\n", A->t_id);
        worker->hasWork = false;
        worker->done = false;
        pthread_mutex_unlock(&worker->mutex);

        PlanetsSoA &b = *A->b;
        int t_id = A->t_id;
        int t_N = A->t_N;
        float *ax = A->t_ax;
        float *ay = A->t_ay;
        float *az = A->t_az;

        int n = b.count;
        {
            ZoneScopedN("compute_accelerations");
            for (int i = t_id; i < n; i += t_N)
            {
                for (int j = i+1; j < n; ++j)
                {
                    float dx = b.x[j] - b.x[i];
                    float dy = b.y[j] - b.y[i];
                    float dz = b.z[j] - b.z[i];
                    float dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
                    float dist = sqrt(dist2);

                    float F = (G * b.mass[i] * b.mass[j]) / dist2;
                    float fx = F * dx / dist;
                    float fy = F * dy / dist;
                    float fz = F * dz / dist;

                    ax[i] += fx / b.mass[i];
                    ay[i] += fy / b.mass[i];
                    az[i] += fz / b.mass[i];
                    az[j] -= fz / b.mass[j];
                    ay[j] -= fy / b.mass[j];
                    ax[j] -= fx / b.mass[j];
                }
            }
        }
        {
            ZoneScopedN("signal_completion");
            pthread_mutex_lock(&worker->mutex);
            // printf("Thread %d finished work and is signaling completion to main thread\n", A->t_id);
            worker->done = true;
            pthread_cond_signal(&worker->cond_done);
            pthread_mutex_unlock(&worker->mutex);
        }
    }
    return NULL;
}

void init_workers(PlanetsSoA &b)
{
    int t_N = NUM_THREADS > b.count ? b.count : NUM_THREADS;
    workers = (Worker *)calloc(t_N, sizeof(Worker));
    for (int i = 0; i < t_N; i++)
    {
        pthread_mutex_init(&workers[i].mutex, NULL);
        pthread_cond_init(&workers[i].cond, NULL);
        pthread_cond_init(&workers[i].cond_done, NULL);
        workers[i].hasWork = false;
        workers[i].exit = false;
        workers[i].args.t_id = i;
        workers[i].args.t_N = t_N;
        pthread_create(&workers[i].thread, NULL, accelerations_thread, &workers[i]);
    }
}

void destroy_workers(PlanetsSoA &b)
{
    int t_N = NUM_THREADS > b.count ? b.count : NUM_THREADS;
    for (int i = 0; i < t_N; i++)
    {
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

void accelerations(PlanetsSoA &b)
{
    int n = b.count;
    int t_N = NUM_THREADS > n ? n : NUM_THREADS;

    float **t_ax = new float *[t_N];
    float **t_ay = new float *[t_N];
    float **t_az = new float *[t_N];

    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = new float[n]();
        t_ay[t] = new float[n]();
        t_az[t] = new float[n]();

        workers[t].args.b = &b;
        workers[t].args.t_ax = t_ax[t];
        workers[t].args.t_ay = t_ay[t];
        workers[t].args.t_az = t_az[t];
    }

    for (int i = 0; i < n; ++i)
    {
        b.ax[i] = 0.0f;
        b.ay[i] = 0.0f;
        b.az[i] = 0.0f;
    }

    // Wake up all workers
    {
        ZoneScopedN("wake_up_workers");
        // printf("====== Wake up all workers =====\n");
        for (int t = 0; t < t_N; ++t)
        {
            pthread_mutex_lock(&workers[t].mutex);
            workers[t].hasWork = true;
            workers[t].done = false;
            pthread_cond_signal(&workers[t].cond);
            pthread_mutex_unlock(&workers[t].mutex);
        }
    }
    {
        ZoneScopedN("wait_for_workers");
        // printf("====== Wait for all workers to be done =====\n");
        for (int t = 0; t < t_N; ++t)
        {
            pthread_mutex_lock(&workers[t].mutex);
            while (!workers[t].done)
                pthread_cond_wait(&workers[t].cond_done, &workers[t].mutex);
            // printf("***** thread %d is done! *****\n", t);
            pthread_mutex_unlock(&workers[t].mutex);
        }
        // printf("===== all threads are done! merging result =====\n");
    }

    {
        ZoneScopedN("merge_results");
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
    }
}
