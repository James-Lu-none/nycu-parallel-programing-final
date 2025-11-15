#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"
#include "tracy/Tracy.hpp"

#include <cstddef>
#include <cmath>

void accelerations_setup(Planet *, std::size_t) {}

void accelerations_teardown() {}

void accelerations(Planet *bodies, std::size_t count)
{
    ZoneScopedN("accelerations_serial");
    if (count == 0) {
        return;
    }

    for (std::size_t i = 0; i < count; ++i) {
        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
    }

    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + config::EPSILON;
            double dist = std::sqrt(dist2);

            double F = (config::G * bodies[i].mass * bodies[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;

            bodies[i].ax += fx / bodies[i].mass;
            bodies[i].ay += fy / bodies[i].mass;
            bodies[i].az += fz / bodies[i].mass;
            bodies[j].ax -= fx / bodies[j].mass;
            bodies[j].ay -= fy / bodies[j].mass;
            bodies[j].az -= fz / bodies[j].mass;
        }
    }
}

