#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

void render(
    canvas &buf,
    const Camera &camera,
    const Planet* bodies,
    const Trail* trails
)
{
    ZoneScopedN("render_serial");
    for (int j = 0; j < HEIGHT; j++)
    {
        for (int i = 0; i < WIDTH; i++)
        {
            vec3 pixel_center = camera.pixel00_loc + (i * camera.pixel_delta_u) + (j * camera.pixel_delta_v);
            vec3 ray_direction = pixel_center - camera.center;
            ray r(camera.center, ray_direction);

            color pixel_color = get_ray_color(r, bodies, trails);
            buf.pixels[j * WIDTH + i] = pixel_color;
        }
    }
}

color get_ray_color(const ray &r, const Planet* bodies, const Trail* trails)
{
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        float t = hit_planet(bodies[i], r);
        if (t >= 0)
        {
            vec3 N = 128 * (unit_vector(r.at(t) - bodies[i].pos) + vec3(1, 1, 1));
            // printf("N: (%f, %f, %f)\n", N.x(), N.y(), N.z());
            return {
                (uint8_t)std::min(N.x(), 255.0),
                (uint8_t)std::min(N.y(), 255.0),
                (uint8_t)std::min(N.z(), 255.0),
                255
            };
            // return bodies[i].col;
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
        float a = dot(r.direction(), r.direction());
        float b = -2.0 * dot(r.direction(), oc);
        float c = dot(oc, oc) - radius_squared;
        float discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            return true;
        }
    }
    return false;
}

float hit_planet(const Planet &p, const ray &r)
{
    vec3 oc = p.pos - r.origin();
    float a = dot(r.direction(), r.direction());
    float b = -2.0 * dot(r.direction(), oc);
    float c = dot(oc, oc) - p.r * p.r;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}