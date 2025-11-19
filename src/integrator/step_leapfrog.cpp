#include "integrator.hpp"
#include "accelerations.hpp"
#include "planet.hpp"

void integrator(Planet b[], double dt)
{
    static int first = 1;

    if (first)
    {
        ZoneScopedN("step_leapfrog_first");
        accelerations(b);
        first = 0;
    }

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        b[i].vel += 0.5 * b[i].acc * dt;
        b[i].pos += b[i].vel * dt;
    }

    accelerations(b);

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        ZoneScopedN("step_leapfrog");
        b[i].vel += 0.5 * b[i].acc * dt;
    }
}
