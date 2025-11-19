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
    const Planet* bodies,
    const Trail* trails
)
{
    for (int j = 0; j < HEIGHT; j++)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            vec3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            vec3 ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = get_ray_color(r, bodies, trails);
            buf.pixels[j * WIDTH + i] = pixel_color;
        }
    }
}

color get_ray_color(const ray &r, const Planet* bodies, const Trail* trails)
{
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        if (hit_planet(bodies[i], r))
        {
            return bodies[i].col;
        }
        if (hit_trail(trails[i], r))
        {
            return bodies[i].col;
        }
    }
    return {0, 0, 0, 255};
}

bool hit_trail(const Trail &t, const ray &r)
{
    int radius_squared = 4; // radius 2
    for (int i = 0; i < t.size; ++i)
    {
        vec3 oc = t.pos[i] - r.origin();
        double a = dot(r.direction(), r.direction());
        double b = -2.0 * dot(r.direction(), oc);
        double c = dot(oc, oc) - radius_squared;
        double discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            return true;
        }
    }
    return false;
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