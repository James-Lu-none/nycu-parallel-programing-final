#include "accelerations.hpp"
#include "planet.hpp"
#include "config.hpp"

typedef struct
{
    PlanetsSoA* b;
    int t_id;
    int t_N;
    float *t_ax;
    float *t_ay;
    float *t_az;
} AccelerationArgs;

static void *accelerations_thread(void *arg)
{
    AccelerationArgs *A = (AccelerationArgs *)arg;
    PlanetsSoA& b = *A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;
    float *ax = A->t_ax;
    float *ay = A->t_ay;
    float *az = A->t_az;

    int n = b.count;
    
    // add t_N-1 to n before divide by t_N to ensure all acceleration pairs (i,j) are covered
    int chunk = (n + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < n) ? (i_start + chunk) : n;

    for (int i = i_start; i < i_end; ++i)
    {
        for (int j = i + 1; j < n; ++j) // Loop for unique pairs (i, j) where i < j
        {
            float dx = b.x[j] - b.x[i];
            float dy = b.y[j] - b.y[i];
            float dz = b.z[j] - b.z[i];
            float dist2 = dx*dx + dy*dy + dz*dz + EPSILON;
            float dist = sqrt(dist2);

            float F = (G * b.mass[i] * b.mass[j]) / dist2;
            float fx = F * dx / dist;
            float fy = F * dy / dist;
            float fz = F * dz / dist;

            // Apply force to body i
            ax[i] += fx / b.mass[i];
            ay[i] += fy / b.mass[i];
            az[i] += fz / b.mass[i];

            // Apply opposite force to body j
            ax[j] -= fx / b.mass[j];
            ay[j] -= fy / b.mass[j];
            az[j] -= fz / b.mass[j];
        }
    }
    return NULL;
}

void accelerations(PlanetsSoA &b)
{
    ZoneScopedN("accelerations");
    int n = b.count;
    int t_N = NUM_THREADS > n ? n : NUM_THREADS;

    pthread_t *threads = new pthread_t[t_N]; // Changed from malloc to new
    AccelerationArgs *args = new AccelerationArgs[t_N]; // Changed from malloc to new
    
    float **t_ax = new float*[t_N];
    float **t_ay = new float*[t_N];
    float **t_az = new float*[t_N];

    // Initialize accelerations to zero for all bodies
    for (int i = 0; i < n; ++i)
    {
        b.ax[i] = 0.0f;
        b.ay[i] = 0.0f;
        b.az[i] = 0.0f;
    }

    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = new float[n](); // Initialize with zeros
        t_ay[t] = new float[n](); // Initialize with zeros
        t_az[t] = new float[n](); // Initialize with zeros

        args[t].b = &b;
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

    // Aggregate results from all threads
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
    delete[] threads;
    delete[] args;
}
