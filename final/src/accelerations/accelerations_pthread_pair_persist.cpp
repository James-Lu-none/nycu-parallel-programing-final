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
        {
            ZoneScopedN("wait_for_work");
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
        }
        
        const Planet *b = A->b;
        int t_id = A->t_id;
        int t_N = A->t_N;
        double *ax = A->t_ax;
        double *ay = A->t_ay;
        double *az = A->t_az;

        int chunk = (NUM_BODIES + t_N - 1) / t_N;
        int i_start = t_id * chunk;
        int i_end = (i_start + chunk < NUM_BODIES) ? (i_start + chunk) : NUM_BODIES;

        {
            ZoneScopedN("compute_accelerations");
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

void init_workers(void)
{
    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;
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

void destroy_workers(void)
{
    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;
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

void accelerations(Planet b[])
{
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
}
