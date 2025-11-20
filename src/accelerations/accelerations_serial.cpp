#include "accelerations.hpp"
#include "planet.hpp"

#include "config.hpp"

void accelerations(Planet b[])
{
    ZoneScopedN("accelerations");
    for (int i = 0; i < NUM_BODIES; ++i)
        b[i].acc = vec3(0.0, 0.0, 0.0);

    for (int i = 0; i < NUM_BODIES; ++i)
    {
        for (int j = i + 1; j < NUM_BODIES; ++j)
        {
            vec3 dpos = b[j].pos - b[i].pos;
            float dist2 = dpos.length_squared() + EPSILON;
            float dist = sqrt(dist2);

            float F = (G * b[i].mass * b[j].mass) / dist2;
            vec3 force = F * dpos / dist;
            b[i].acc += force / b[i].mass;
            b[j].acc -= force / b[j].mass;
        }
    }
}
