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
        for (int i = 0; i < n; ++i)
        {
            b[i].vel += 0.5f * b[i].acc * dt;
            b[i].pos += b[i].vel * dt;
        }
    }

    // Drift
    {
        // First half kick
        ZoneScopedN("step_leapfrog 1/2");
        for (int i = 0; i < n; ++i)
        {
            b[i].vel += 0.5f * b[i].acc * dt;
            b[i].pos += b[i].vel * dt;
        }
    }

    accelerations(b);

    // Second half kick
    {
        ZoneScopedN("step_leapfrog 2/2");
        for (int i = 0; i < n; ++i)
        {
            b[i].vel += 0.5f * b[i].acc * dt;
            b[i].pos += b[i].vel * dt;
        }
    }
#endif
}
