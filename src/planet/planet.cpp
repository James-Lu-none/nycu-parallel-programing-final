#include "planet.hpp"
#include "vec3.hpp"

vec3 get_center_of_mass(Planet b[])
{
    ZoneScopedN("recenter");
    vec3 com_pos(0.0, 0.0, 0.0);
    float total_mass = 0.0;
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        com_pos += b[i].mass * b[i].pos;
        total_mass += b[i].mass;
    }
    com_pos = com_pos / total_mass;
    return com_pos;
}

void trail_push(Trail *t, vec3 pos)
{
    ZoneScopedN("trail_push");
    if (t->size == 0)
    {
        t->pos[0] = pos;
        t->head = 1 % TRAIL_BUF;
        t->size = 1;
        return;
    }
    int last = (t->head - 1 + TRAIL_BUF) % TRAIL_BUF;
    if ((pos - t->pos[last]).length() >= MIN_DIST)
    {
        t->pos[t->head] = pos;
        t->head = (t->head + 1) % TRAIL_BUF;
        if (t->size < TRAIL_BUF)
            t->size++;
    }
}