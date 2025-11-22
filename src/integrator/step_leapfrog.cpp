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

    if (first)
    {
        ZoneScopedN("step_leapfrog_first");
        accelerations(b);
        first = 0;
    }

    for (int i = 0; i < n; ++i)
    {
        b[i].vel += 0.5 * b[i].acc * dt;
        b[i].pos += b[i].vel * dt;
    }

    accelerations(b);

    for (int i = 0; i < n; ++i)
    {
        ZoneScopedN("step_leapfrog");
        b[i].vel += 0.5 * b[i].acc * dt;
    }
#endif
}
