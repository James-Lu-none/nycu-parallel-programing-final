#include "integrator.hpp"
#include "accelerations.hpp"
#include "planet.hpp"

void integrator(vector<Planet>& b, float dt)
{
    static const int n = b.size();
    static int first = 1;

    if (first)
    {
        ZoneScopedN("step_leapfrog_first");
        accelerations(b);
        first = 0;
    }

    {
        ZoneScopedN("step_leapfrog 1/2");
        for (int i = 0; i < n; ++i)
        {
            b[i].vel += 0.5 * b[i].acc * dt;
            b[i].pos += b[i].vel * dt;
        }
    }

    accelerations(b);
    {
        ZoneScopedN("step_leapfrog 2/2");
        for (int i = 0; i < n; ++i)
        {
            b[i].vel += 0.5 * b[i].acc * dt;
        }
    }
}
