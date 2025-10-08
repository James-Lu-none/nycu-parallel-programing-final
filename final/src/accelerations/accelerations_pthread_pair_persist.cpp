#include "config.hpp"

typedef struct
{
    const Planet *b;
    int t_id;
    int t_N;
    double *t_ax;
    double *t_ay;
    double *t_az;
} AccelerationArgs;

#define NUM_THREADS 4 // or whatever you had

typedef struct
{
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool hasWork;
    bool exit;
    AccelerationArgs args;
} Worker;

static Worker *workers = NULL;

static void *accelerations_thread(void *arg)
{
    Worker *worker = (Worker *)arg;
    AccelerationArgs *A = &worker->args;
    
    while (1)
    {
        pthread_mutex_lock(&worker->mutex);
        while (!worker->hasWork && !worker->exit)
            pthread_cond_wait(&worker->cond, &worker->mutex);
        bool shouldExit = worker->exit;
        worker->hasWork = false;
        pthread_mutex_unlock(&worker->mutex);

        if (shouldExit)
            break;

        const Planet *b = A->b;
        int t_id = A->t_id;
        int t_N = A->t_N;
        double *ax = A->t_ax;
        double *ay = A->t_ay;
        double *az = A->t_az;

        int chunk = (NUM_BODIES + t_N - 1) / t_N;
        int i_start = t_id * chunk;
        int i_end = (i_start + chunk < NUM_BODIES) ? (i_start + chunk) : NUM_BODIES;

        for (int i = i_start; i < i_end; ++i)
        {
            for (int j = 0; j < NUM_BODIES; ++j)
            {
                double dx = b[j].x - b[i].x;
                double dy = b[j].y - b[i].y;
                double dz = b[j].z - b[i].z;
                double dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
                double dist = sqrt(dist2);

                double F = (G * b[i].mass * b[j].mass) / dist2;
                double fx = F * dx / dist;
                double fy = F * dy / dist;
                double fz = F * dz / dist;

                ax[i] += fx / b[i].mass;
                ay[i] += fy / b[i].mass;
                az[i] += fz / b[i].mass;
            }
        }
    }
    return NULL;
}

void init_workers(void)
{
    workers = (Worker *)calloc(NUM_THREADS, sizeof(Worker));
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_mutex_init(&workers[i].mutex, NULL);
        pthread_cond_init(&workers[i].cond, NULL);
        workers[i].hasWork = false;
        workers[i].exit = false;
        workers[i].args.t_id = i;
        workers[i].args.t_N = NUM_THREADS;
        pthread_create(&workers[i].thread, NULL, accelerations_thread, &workers[i]);
    }
}

void destroy_workers(void)
{
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].exit = true;
        pthread_cond_signal(&workers[i].cond);
        pthread_mutex_unlock(&workers[i].mutex);
        pthread_join(workers[i].thread, NULL);
        pthread_mutex_destroy(&workers[i].mutex);
        pthread_cond_destroy(&workers[i].cond);
    }
    free(workers);
    workers = NULL;
}

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations_parallel");

    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;

    double **t_ax = (double **)malloc(sizeof(double *) * t_N);
    double **t_ay = (double **)malloc(sizeof(double *) * t_N);
    double **t_az = (double **)malloc(sizeof(double *) * t_N);
    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_ay[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_az[t] = (double *)calloc(NUM_BODIES, sizeof(double));

        workers[t].args.b = b;
        workers[t].args.t_ax = t_ax[t];
        workers[t].args.t_ay = t_ay[t];
        workers[t].args.t_az = t_az[t];
    }

    for (int i = 0; i < NUM_BODIES; ++i)
        b[i].ax = b[i].ay = b[i].az = 0.0;

    // Wake up all workers
    for (int t = 0; t < t_N; ++t)
    {
        pthread_mutex_lock(&workers[t].mutex);
        workers[t].hasWork = true;
        pthread_cond_signal(&workers[t].cond);
        pthread_mutex_unlock(&workers[t].mutex);
    }

    bool done;
    do
    {
        done = true;
        for (int t = 0; t < t_N; ++t)
        {
            pthread_mutex_lock(&workers[t].mutex);
            if (workers[t].hasWork)
                done = false;
            pthread_mutex_unlock(&workers[t].mutex);
        }
        if (!done)
            sched_yield();
    } while (!done);

    // Merge results
    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < NUM_BODIES; ++i)
        {
            b[i].ax += t_ax[t][i];
            b[i].ay += t_ay[t][i];
            b[i].az += t_az[t][i];
        }
        free(t_ax[t]);
        free(t_ay[t]);
        free(t_az[t]);
    }

    free(t_ax);
    free(t_ay);
    free(t_az);
}
