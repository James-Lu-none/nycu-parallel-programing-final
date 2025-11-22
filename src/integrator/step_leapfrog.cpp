#include "integrator.hpp"
#include "accelerations.hpp"
#include "planet.hpp"

void integrator(PlanetsSoA& b, float dt)
{
    static const int n = b.count;
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
    for (int i = 0; i < n; ++i)
    {
        b.x[i] += b.vx[i] * dt;
        b.y[i] += b.vy[i] * dt;
        b.z[i] += b.vz[i] * dt;
    }

    accelerations(b);

    // Second half kick
    for (int i = 0; i < n; ++i)
    {
        b.vx[i] += 0.5f * b.ax[i] * dt;
        b.vy[i] += 0.5f * b.ay[i] * dt;
        b.vz[i] += 0.5f * b.az[i] * dt;
    }
}
