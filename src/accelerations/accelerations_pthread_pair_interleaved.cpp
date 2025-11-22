#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

typedef struct
{
    vector<Planet> & b;
    int t_id;
    int t_N;
    vec3 *t_acc;
} AccelerationArgs;

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    int n = A->b.size();
    vector<Planet> &b = A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;

    vec3 *acc = A->t_acc;

    int count = 0;

    for (int i = t_id; i < n; i+=t_N)
    {
        for (int j = i + 1; j < n; ++j)
        {
            vec3 dpos = b[j].pos - b[i].pos;
            float dist2 = dpos.length_squared() + EPSILON;
            float dist = sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;

            acc[i] += force / b[i].mass;
            acc[j] -= force / b[j].mass;
            count++;
        }
    }
    printf("Thread %d: calc_count=%d\n", t_id, count);
    return NULL;
}

void accelerations(vector<Planet> &b)
{
    ZoneScopedN("accelerations_parallel");

    int n = b.size();
    int t_N = NUM_THREADS > n ? n : NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    AccelerationArgs *args = (AccelerationArgs *)malloc(sizeof(AccelerationArgs) * t_N);

    vec3 **t_acc = new vec3*[t_N];
    for (int t = 0; t < t_N; ++t)
    {
        t_acc[t] = new vec3[n]();
    }

    for (int i = 0; i < n; ++i)
        b[i].acc = vec3(0.0, 0.0, 0.0);

    for (int t = 0; t < t_N; ++t)
    {
        args[t].b = b;
        args[t].t_id = t;
        args[t].t_N = t_N;
        args[t].t_acc = t_acc[t];
        pthread_create(&threads[t], NULL, accelerations_thread, &args[t]);
    }

    for (int t = 0; t < t_N; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < n; ++i)
        {
            b[i].acc += t_acc[t][i];
        }
    }

    for (int t = 0; t < t_N; ++t)
    {
        delete[] t_acc[t];
    }
    delete[] t_acc;
    free(threads);
    free(args);
}
