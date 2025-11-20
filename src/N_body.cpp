#include "accelerations.hpp"
#include "planet.hpp"
#include "render.hpp"
#include "integrator.hpp"
#include "camera.hpp"

#include "config.hpp"

using namespace std;

canvas canvas_buf;

vec3 random_vec3(float min, float max)
{
    ZoneScopedN("random_float");
    float range = (max - min);
    float div = RAND_MAX / range;
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

    if (!surf) {
        fprintf(stderr, "SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    const unsigned long colors[] = {0x00ff0000, 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff};
    Planet bodies[NUM_BODIES];
    const float S  = 140.0;
    const float VS = 140.0;
    const float m  = 200.0;

    Camera camera = Camera();

    for (int i = 0; i < NUM_BODIES; ++i){
        bodies[i] = (Planet){
            random_vec3(-1.0, 1.0) * S,
            random_vec3(-1.0, 1.0) * VS,
            vec3(0.0, 0.0, 0.0),
            m, 15,
            {
                (uint8_t)((colors[i % 6] >> 16) & 0xFF),
                (uint8_t)((colors[i % 6] >> 8) & 0xFF),
                (uint8_t)(colors[i % 6] & 0xFF),
                255
            }};
    }

    Trail trails[NUM_BODIES] = {};

    int running = 1;
    SDL_Event ev;
    const float FIXED_DT = 0.0002;
    float accumulator = 0.0;
    Uint32 prev = SDL_GetTicks();
    
    // For tracking average frame time
    float total_frame_time = 0.0;
    int frame_count = 0;
    
    #ifdef INIT_REQUIRED
        init_workers();
    #endif
    while (running)
    {
        ZoneScopedN("MainLoop");
        FrameMark;
        
        {
            ZoneScopedN("EventHandling");
            while (SDL_PollEvent(&ev))
            {
                camera.handle_event(ev);
                if (ev.type == SDL_QUIT)
                    running = 0;
            }
            camera.update_view(bodies);
        }
        
        Uint32 now = SDL_GetTicks();
        float frame_dt = (now - prev) / 1000.0;
        prev = now;
        if (frame_dt > 0.05) frame_dt = 0.05;
        accumulator += frame_dt;

        {
            ZoneScopedN("PhysicsStep");
            while (accumulator >= FIXED_DT) {
                integrator(bodies, FIXED_DT);
                accumulator -= FIXED_DT;
            }
        }

        {
            ZoneScopedN("TrailUpdate");
            for (int i = 0; i < NUM_BODIES; ++i)
                trail_push(&trails[i], bodies[i].pos);
        }
        
        {
            ZoneScopedN("RenderStep");
            SDL_LockSurface(surf);
            render(
                surf->pixels,
                camera,
                bodies,
                trails
            );
            SDL_UnlockSurface(surf);
        }

        {
            ZoneScopedN("DisplayUpdate");
            SDL_BlitSurface(surf, NULL, win_surf, NULL);
            SDL_UpdateWindowSurface(win);
        }
        
        // Track frame time
        frame_count++;
        total_frame_time += frame_dt;
        float avg_frame_time = total_frame_time / frame_count;
        
        TracyPlot("Frame Time (ms)", frame_dt * 1000.0);
        TracyPlot("Average Frame Time (ms)", avg_frame_time * 1000.0);
        
        SDL_Delay(16);
    }
    #ifdef INIT_REQUIRED
        destroy_workers();
    #endif
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}