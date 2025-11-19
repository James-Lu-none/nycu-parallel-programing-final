#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

void render(
    canvas &buf, 
    const point3 &camera_center,
    const vec3 &pixel00_loc,
    const vec3 &pixel_delta_u,
    const vec3 &pixel_delta_v
)
{
    for (int j = 0; j < HEIGHT; j++)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            vec3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            vec3 ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = get_ray_color(r);
            buf.pixels[j * WIDTH + i] = pixel_color;
        }
    }
}

color get_ray_color(const ray &r)
{
    return {255, 255, 255};
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