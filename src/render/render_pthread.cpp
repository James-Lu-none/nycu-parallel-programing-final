#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

typedef struct {
    void* buf;
    const Camera* camera;
    const Planet* bodies;
    const Trail* trails;
    int start_row;
    int end_row;
} RenderTaskArgs;

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
                (uint8_t)std::min(N.x(), (float)255.0),
                (uint8_t)std::min(N.y(), (float)255.0),
                (uint8_t)std::min(N.z(), (float)255.0),
                255
            };
            // return bodies[i].col;
        }
        // if (hit_trail(trails[i], r))
        // {
        //     return bodies[i].col;
        // }
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

void *render_thread(void *args_void){
    RenderTaskArgs *args = (RenderTaskArgs *)args_void;
    const int start_row = args->start_row;
    const int end_row = args->end_row;
    color *buf = (color *)malloc(sizeof(color) * WIDTH * (end_row - start_row));
    const Camera *camera = args->camera;
    const Planet *bodies = args->bodies;
    const Trail *trails = args->trails;

    for (int j = start_row; j < end_row; ++j){
        for (int i = 0; i < WIDTH; ++i){
            float u = float(i) / (WIDTH - 1);
            float v = float(j) / (HEIGHT - 1);
            vec3 pixel_center = camera->pixel00_loc + (i * camera->pixel_delta_u) + (j * camera->pixel_delta_v);
            vec3 ray_direction = pixel_center - camera->center;
            ray r(camera->center, ray_direction);

            color pixel_color = get_ray_color(r, bodies, trails);
            buf[(j - start_row) * WIDTH + i] = pixel_color;
        }
    }

    memcpy(
        (color*)args->buf + (start_row * WIDTH),
        buf,
        sizeof(color) * WIDTH * (end_row - start_row)
    );
    return NULL;
}


void render(
    void *buf,
    const Camera &camera,
    const Planet* bodies,
    const Trail* trails
)
{
    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    RenderTaskArgs *args = (RenderTaskArgs *)malloc(sizeof(RenderTaskArgs) * t_N);

    int rows_per_thread = HEIGHT / t_N;
    for (int i = 0; i < t_N; ++i){
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i == t_N - 1) ? HEIGHT : (i + 1) * rows_per_thread;
        args[i].buf = &buf;
        args[i].camera = &camera;
        args[i].bodies = bodies;
        args[i].trails = trails;
        pthread_create(&threads[i], NULL, render_thread, (void *)&args[i]);
    }
    for (int i = 0; i < t_N; ++i){
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(args);
}