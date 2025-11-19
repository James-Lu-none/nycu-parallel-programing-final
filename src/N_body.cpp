#include "accelerations.hpp"
#include "planet.hpp"
#include "render.hpp"
#include "integrator.hpp"

#include "config.hpp"

using namespace std;

canvas canvas_buf;

vec3 random_vec3(double min, double max)
{
    ZoneScopedN("random_double");
    double range = (max - min);
    double div = RAND_MAX / range;
    return vec3(
        min + (rand() / div),
        min + (rand() / div),
        min + (rand() / div)
    );
}

int main(void)
{
    tracy::SetThreadName("main_thread");
    srand(time(NULL));
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("Three-Body Problem",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Surface *win_surf = SDL_GetWindowSurface(win);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Surface *surf = SDL_CreateRGBSurfaceWithFormat(
        0,
        WIDTH, HEIGHT,
        32,
        SDL_PIXELFORMAT_ARGB8888
    );
    printf("pitch: %d bytes\n", surf->pitch);

    if (!surf) {
        fprintf(stderr, "SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    const unsigned long colors[] = {0x00ff0000, 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff};
    Planet bodies[NUM_BODIES];
    const double S  = 140.0;
    const double VS = 140.0;
    const double m  = 200.0;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(WIDTH) / HEIGHT);
    point3 camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, viewport_height, 0);
    vec3 pixel_delta_u = viewport_u / WIDTH;
    vec3 pixel_delta_v = viewport_v / HEIGHT;

    vec3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    for (int i = 0; i < NUM_BODIES; ++i){
        bodies[i] = (Planet){
            random_vec3(-1.0, 1.0) * S,
            random_vec3(-1.0, 1.0) * VS,
            vec3(0.0, 0.0, 0.0),
            m, 15};
    }

    Trail trails[NUM_BODIES] = {};
    
    int running = 1;
    SDL_Event ev;
    const double FIXED_DT = 0.0002;
    double accumulator = 0.0;
    Uint32 prev = SDL_GetTicks();
    #ifdef INIT_REQUIRED
        init_workers();
    #endif
    while (running)
    {
        FrameMark;
        while (SDL_PollEvent(&ev))
            if (ev.type == SDL_QUIT) running = 0;

        Uint32 now = SDL_GetTicks();
        double frame_dt = (now - prev) / 1000.0;
        prev = now;
        if (frame_dt > 0.05) frame_dt = 0.05;
        accumulator += frame_dt;

        while (accumulator >= FIXED_DT) {
            integrator(bodies, FIXED_DT);
            accumulator -= FIXED_DT;
        }

        // recenter(bodies);
        for (int i = 0; i < NUM_BODIES; ++i)
            trail_push(&trails[i], bodies[i].pos);
        
        render(
            canvas_buf,
            camera_center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            bodies
        );

        SDL_LockSurface(surf);
        memcpy(surf->pixels, canvas_buf.pixels, WIDTH * HEIGHT * sizeof(color));
        SDL_UnlockSurface(surf);

        SDL_BlitSurface(surf, NULL, win_surf, NULL);
        SDL_UpdateWindowSurface(win);
        SDL_Delay(16);
    }
    #ifdef INIT_REQUIRED
        destroy_workers();
    #endif
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}