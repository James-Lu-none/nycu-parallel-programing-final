#include "accelerations.hpp"
#include "planet.hpp"
#include "render.hpp"
#include "integrator.hpp"

#include "config.hpp"

using namespace std;

double random_double(double min, double max)
{
    ZoneScopedN("random_double");
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
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
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Surface *surf = SDL_GetWindowSurface(win);
    if (!surf) {
        fprintf(stderr, "SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    const unsigned long colors[] = {0x00ff0000, 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff};
    Planet bodies[NUM_BODIES];
    const double S  = 140.0;
    const double VS = 140.0;
    const double m  = 200.0;
    double cx = WIDTH / 2.0;
    double cy = HEIGHT / 2.0;
    double cz = 0.0;

    for (int i = 0; i < NUM_BODIES; ++i){
        bodies[i] = (Planet){
            cx + random_double(-1.0, 1.0) * S,
            cy + random_double(-1.0, 1.0) * S,
            cz + random_double(-0.1, 0.1) * S,
            random_double(-1.0, 1.0) * VS,
            random_double(-1.0, 1.0) * VS,
            random_double(-0.1, 0.1) * VS,
            0.0, 0.0, 0.0,
            m, 15};
    }

    Trail trails[NUM_BODIES] = {0};

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

        recenter(bodies);

        for (int i = 0; i < NUM_BODIES; ++i)
            trail_push(&trails[i], (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z);

        SDL_FillRect(surf, NULL, COL_BLACK);
        
        for (int i = 0; i < NUM_BODIES; ++i) {
            trail_draw(surf, &trails[i], colors[i % 6]);
            fill_circle(surf, (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z, (int) bodies[i].r, colors[i % 6]);
        }

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