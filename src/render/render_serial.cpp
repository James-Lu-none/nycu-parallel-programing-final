#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

void render(
    canvas &buf, 
    const point3 &camera_center,
    const vec3 &pixel00_loc,
    const vec3 &pixel_delta_u,
    const vec3 &pixel_delta_v,
    const Planet* bodies
)
{
    for (int j = 0; j < HEIGHT; j++)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            vec3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            vec3 ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = get_ray_color(r, bodies);
            buf.pixels[j * WIDTH + i] = pixel_color;
        }
    }
}

color get_ray_color(const ray &r, const Planet* bodies)
{
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        if (hit_planet(bodies[i], r))
        {
            return bodies[i].col;
        }
    }
    return {0, 0, 0, 255};
}

bool hit_planet(const Planet &p, const ray &r)
{
    vec3 oc = p.pos - r.origin();
    double a = dot(r.direction(), r.direction());
    double b = -2.0 * dot(r.direction(), oc);
    double c = dot(oc, oc) - p.r * p.r;
    double discriminant = b * b - 4 * a * c;
    return (discriminant >= 0);
}

void recenter(Planet b[])
{
    ZoneScopedN("recenter");
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