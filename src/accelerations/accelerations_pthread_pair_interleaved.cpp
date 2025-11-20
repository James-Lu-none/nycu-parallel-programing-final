#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

typedef struct
{
    const Planet *b;
    int t_id;
    int t_N;
    float *t_ax;
    float *t_ay;
    float *t_az;
} AccelerationArgs;

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    const Planet *b = A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;

    float *ax = A->t_ax;
    float *ay = A->t_ay;
    float *az = A->t_az;

    int count = 0;

    for (int i = t_id; i < NUM_BODIES; i+=t_N)
    {
        for (int j = i + 1; j < NUM_BODIES; ++j)
        {
            float dx = b[j].x - b[i].x;
            float dy = b[j].y - b[i].y;
            float dz = b[j].z - b[i].z;
            float dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
            float dist = sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            float fx = F * dx / dist;
            float fy = F * dy / dist;
            float fz = F * dz / dist;

            ax[i] += fx / b[i].mass;
            ay[i] += fy / b[i].mass;
            az[i] += fz / b[i].mass;
            ax[j] -= fx / b[j].mass;
            ay[j] -= fy / b[j].mass;
            az[j] -= fz / b[j].mass;
            count++;
        }
    }
    printf("Thread %d: calc_count=%d\n", t_id, count);
    return NULL;
}

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations_parallel");

    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    AccelerationArgs *args = (AccelerationArgs *)malloc(sizeof(AccelerationArgs) * t_N);

    float **t_ax = (float **)malloc(sizeof(float *) * t_N);
    float **t_ay = (float **)malloc(sizeof(float *) * t_N);
    float **t_az = (float **)malloc(sizeof(float *) * t_N);
    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = (float *)calloc(NUM_BODIES, sizeof(float));
        t_ay[t] = (float *)calloc(NUM_BODIES, sizeof(float));
        t_az[t] = (float *)calloc(NUM_BODIES, sizeof(float));
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
        pthread_create(&threads[t], NULL, accelerations_thread, &args[t]);
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
