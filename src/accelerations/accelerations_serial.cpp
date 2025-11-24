#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

void accelerations(PlanetsSoA& b)
{
    ZoneScopedN("accelerations");

    int n = b.count;
    for (int i = 0; i < n; ++i)
    {
        b.ax[i] = 0.0f;
        b.ay[i] = 0.0f;
        b.az[i] = 0.0f;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            float dx = b.x[j] - b.x[i];
            float dy = b.y[j] - b.y[i];
            float dz = b.z[j] - b.z[i];
            
            float dist2 = dx*dx + dy*dy + dz*dz + EPSILON;
            float dist = std::sqrt(dist2);

            float F = (G * b.mass[i] * b.mass[j]) / dist2;
            float fx = F * dx / dist;
            float fy = F * dy / dist;
            float fz = F * dz / dist;

            b.ax[i] += fx / b.mass[i];
            b.ay[i] += fy / b.mass[i];
            b.az[i] += fz / b.mass[i];

            b.ax[j] -= fx / b.mass[j];
            b.ay[j] -= fy / b.mass[j];
            b.az[j] -= fz / b.mass[j];
        }
    }
}
