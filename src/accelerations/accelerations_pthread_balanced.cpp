#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

typedef struct
{
    const vector<Planet>* b;
    int t_id;
    int t_N;
    vec3 *t_acc;
} AccelerationArgs;

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    const vector<Planet> &b = *A->b;
    int n = b.size();
    int t_id = A->t_id;
    int t_N = A->t_N;

    vec3 *acc = A->t_acc;

    // add t_N-1 to n before divide by t_N to ensure all acceleration pairs (i,j) are covered
    int chunk = (n + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < n) ? (i_start + chunk) : n;
    int count = 0;

    for (int i = i_start; i < i_end; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            vec3 dpos = b[j].pos - b[i].pos;
            float dist2 = dpos.length_squared() + EPSILON;
            float dist = sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;
            acc[i] += force / b[i].mass;
            count++;
        }
    }
    // printf("Thread %d: i_start=%d, i_end=%d, calc_count=%d\n", t_id, i_start, i_end, count);
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
        args[t].b = &b;
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

    // for (int i = 0; i < n; ++i)
    // {
    //     printf("[debug] Body %d Acceleration: %f %f %f\n", i, b[i].acc.x(), b[i].acc.y(), b[i].acc.z());
    // }
    free(threads);
    free(args);
}
