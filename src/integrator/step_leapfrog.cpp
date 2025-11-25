#include "integrator.hpp"
#include "accelerations.hpp"
#include "planet.hpp"

#ifdef USE_CUDA
void accelerations_integrate(Planet b[], int n, float dt);
#endif

void integrator(vector<Planet>& b, float dt)
{
#ifdef USE_CUDA
    accelerations_integrate(b.data(), b.size(), dt);
#else
    static const int n = b.size();
    static int first = 1;

    if (first)
    {
        accelerations(b);
        first = 0;
        // First half kick
        for (int i = 0; i < n; ++i)
        {
            b.vx[i] += 0.5f * b.ax[i] * dt;
            b.vy[i] += 0.5f * b.ay[i] * dt;
            b.vz[i] += 0.5f * b.az[i] * dt;
        }
    }

    // Drift
    {
        ZoneScopedN("step_leapfrog 1/2");
        for (int i = 0; i < n; ++i)
        {
            b.x[i] += b.vx[i] * dt;
            b.y[i] += b.vy[i] * dt;
            b.z[i] += b.vz[i] * dt;
        }
    }

    accelerations(b);

    // Second half kick
    {
        ZoneScopedN("step_leapfrog 2/2");
        for (int i = 0; i < n; ++i)
        {
            b.vx[i] += 0.5f * b.ax[i] * dt;
            b.vy[i] += 0.5f * b.ay[i] * dt;
            b.vz[i] += 0.5f * b.az[i] * dt;
        }
    }
#endif
}
