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
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
        b[i].x += b[i].vx * dt;
        b[i].y += b[i].vy * dt;
        b[i].z += b[i].vz * dt;
    }

    accelerations(b);

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        ZoneScopedN("step_leapfrog");
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
    }
}
