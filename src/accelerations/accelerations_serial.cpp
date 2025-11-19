#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations");
    for (int i = 0; i < NUM_BODIES; ++i)
        b[i].ax = b[i].ay = b[i].az = 0.0;

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        for (int j = i + 1; j < NUM_BODIES; ++j)
        {
            double dx = b[j].x - b[i].x;
            double dy = b[j].y - b[i].y;
            double dz = b[j].z - b[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
            double dist = sqrt(dist2);

            double F = (G * b[i].mass * b[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;

            b[i].ax += fx / b[i].mass;
            b[i].ay += fy / b[i].mass;
            b[i].az += fz / b[i].mass;
            b[j].ax -= fx / b[j].mass;
            b[j].ay -= fy / b[j].mass;
            b[j].az -= fz / b[j].mass;
        }
    }
}
