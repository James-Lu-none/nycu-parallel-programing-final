#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

typedef struct
{
    void *buf;
    const Camera *camera;
    const vector<Planet> *bodies;
    const Trail *trails;
    int start_row;
    int end_row;
} RenderTaskArgs;

void *render_thread(void *args_void)
{
    ZoneScopedN("rendering thread");
    RenderTaskArgs *args = (RenderTaskArgs *)args_void;
    const int start_row = args->start_row;
    const int end_row = args->end_row;
    color *buf = (color *)malloc(sizeof(color) * WIDTH * (end_row - start_row));
    const Camera *camera = args->camera;
    const vector<Planet> *bodies = args->bodies;
    const Trail *trails = args->trails;

    for (int j = start_row; j < end_row; ++j)
    {
        for (int i = 0; i < WIDTH; ++i)
        {
            float u = float(i) / (WIDTH - 1);
            float v = float(j) / (HEIGHT - 1);
            vec3 pixel_center = camera->pixel00_loc + (i * camera->pixel_delta_u) + (j * camera->pixel_delta_v);
            vec3 ray_direction = pixel_center - camera->center;
            ray r(camera->center, ray_direction);

            color pixel_color = get_ray_color_simd(r, *bodies, trails);
            buf[(j - start_row) * WIDTH + i] = pixel_color;
        }
    }

    memcpy(
        (color *)args->buf + (start_row * WIDTH),
        buf,
        sizeof(color) * WIDTH * (end_row - start_row));
    return NULL;
}

void render(
    uint32_t *pixels,
    const Camera &camera,
    const vector<Planet> &bodies,
    const Trail *trails
)
{
    int t_N = config::NUM_THREADS > bodies.size() ? bodies.size() : config::NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    RenderTaskArgs *args = (RenderTaskArgs *)malloc(sizeof(RenderTaskArgs) * t_N);

    int rows_per_thread = HEIGHT / t_N;
    for (int i = 0; i < t_N; ++i)
    {
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i == t_N - 1) ? HEIGHT : (i + 1) * rows_per_thread;
        args[i].buf = pixels;
        args[i].camera = &camera;
        args[i].bodies = &bodies;
        args[i].trails = trails;
        pthread_create(&threads[i], NULL, render_thread, (void *)&args[i]);
    }
    for (int i = 0; i < t_N; ++i)
    {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(args);
}