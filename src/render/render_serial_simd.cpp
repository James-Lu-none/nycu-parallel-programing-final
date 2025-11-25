#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

void render(
    uint32_t *pixels,
    const Camera &camera,
    const vector<Planet> &bodies,
    const Trail *trails
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

            color pixel_color = get_ray_color_simd(r, bodies, trails);
            ((color*)pixels)[j * WIDTH + i] = pixel_color;
        }
    }
}
