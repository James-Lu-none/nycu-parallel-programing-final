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

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    const Planet *b = A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;

    double *ax = A->t_ax;
    double *ay = A->t_ay;
    double *az = A->t_az;

    // add t_N-1 to NUM_BODIES before divide by t_N to ensure all acceleration pairs (i,j) are covered
    int chunk = (NUM_BODIES + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < NUM_BODIES) ? (i_start + chunk) : NUM_BODIES;
    int count = 0;

    for (int i = i_start; i < i_end; ++i)
    {
        for (int j = i + 1; j < NUM_BODIES; ++j)
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
            ax[j] -= fx / b[j].mass;
            ay[j] -= fy / b[j].mass;
            az[j] -= fz / b[j].mass;
            count++;
        }
    }
    printf("Thread %d: i_start=%d, i_end=%d, calc_count=%d\n", t_id, i_start, i_end, count);
    return NULL;
}

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations_parallel");

    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    AccelerationArgsV1 *args = (AccelerationArgsV1 *)malloc(sizeof(AccelerationArgsV1) * t_N);

    double **t_ax = (double **)malloc(sizeof(double *) * t_N);
    double **t_ay = (double **)malloc(sizeof(double *) * t_N);
    double **t_az = (double **)malloc(sizeof(double *) * t_N);
    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_ay[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_az[t] = (double *)calloc(NUM_BODIES, sizeof(double));
    }

    for (int i = 0; i < NUM_BODIES; ++i)
        b[i].ax = b[i].ay = b[i].az = 0.0;

    for (int t = 0; t < t_N; ++t)
    {
        args[t].b = b;
        args[t].t_id = t;
        args[t].t_N = t_N;
        args[t].t_ax = t_ax[t];
        args[t].t_ay = t_ay[t];
        args[t].t_az = t_az[t];
        pthread_create(&threads[t], NULL, accelerations_thread_v2, &args[t]);
    }

    for (int t = 0; t < t_N; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < NUM_BODIES; ++i)
        {
            b[i].ax += t_ax[t][i];
            b[i].ay += t_ay[t][i];
            b[i].az += t_az[t][i];
        }
    }

    for (int t = 0; t < t_N; ++t)
    {
        free(t_ax[t]);
        free(t_ay[t]);
        free(t_az[t]);
    }
    free(t_ax);
    free(t_ay);
    free(t_az);
    free(threads);
    free(args);
}
