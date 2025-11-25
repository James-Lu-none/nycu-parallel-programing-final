#include "accelerations.hpp"
#include "planet.hpp"
#include "render.hpp"
#include "integrator.hpp"
#include "camera.hpp"

#include "config.hpp"

canvas canvas_buf;

namespace config
{
    int NUM_THREADS = 0;
}

int main(int argc, char* argv[])
{
    tracy::SetThreadName("main_thread");
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "N-Body Problem",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, SDL_WINDOW_SHOWN
    );
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

    vector<Planet> bodies;
    PlanetsSoA bodies_soa;
    load_planets_from_file(argc > 1 ? argv[1] : nullptr, bodies);
    load_planets_to_SoA(bodies, bodies_soa);

    config::NUM_THREADS = atoi(argv[2]);

    printf("Using %d threads\n", config::NUM_THREADS);
    if (config::NUM_THREADS <= 0) {
        fprintf(stderr, "Invalid number of threads: %d\n", config::NUM_THREADS);
        return 1; 
    }
    
    Camera camera = Camera();

    bool running = true;
    SDL_Event ev;
    const float FIXED_DT = 0.0002;
    float accumulator = 0.0;
    Uint32 prev = SDL_GetTicks();
    
    // For tracking average frame time
    float total_frame_time = 0.0;
    int frame_count = 0;
    
    #ifdef INIT_REQUIRED
        init_workers(bodies);
    #endif
    #ifdef RENDER_INIT_REQUIRED
        init_render_workers();
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
                    running = false;
            }
            camera.update_view(bodies_soa);
        }
        
        Uint32 now = SDL_GetTicks();
        float frame_dt = (now - prev) / 1000.0;
        prev = now;
        frame_count++;
        total_frame_time += frame_dt;
        float avg_frame_time = total_frame_time / frame_count;
        // Track frame time
        TracyPlot("Frame Time (ms)", frame_dt * 1000.0);
        TracyPlot("Average Frame Time (ms)", avg_frame_time * 1000.0);

        // prevent accumulator adds to much when frame_dt is too large, aka. frame rate is too low (lower then 20 FPS)
        if (frame_dt > 0.05) frame_dt = 0.05;
        accumulator += frame_dt;

        {
            ZoneScopedN("PhysicsStep");
            while (accumulator >= FIXED_DT) {
                integrator(bodies_soa, FIXED_DT);
                accumulator -= FIXED_DT;
            }
        }

        // {
        //     ZoneScopedN("TrailUpdate");
        //     for (int i = 0; i < NUM_BODIES; ++i)
        //         trail_push(&trails[i], bodies[i].pos);
        // }
        
        {
            ZoneScopedN("RenderStep");
            SDL_LockSurface(surf);
            render(
                (uint32_t *)surf->pixels,
                camera,
                bodies_soa,
                nullptr
            );
            SDL_UnlockSurface(surf);
        }

        {
            ZoneScopedN("DisplayUpdate");
            SDL_BlitSurface(surf, NULL, win_surf, NULL);
            SDL_UpdateWindowSurface(win);
        }
    }
    #ifdef INIT_REQUIRED
        destroy_workers(bodies);
    #endif
    #ifdef RENDER_INIT_REQUIRED
        destroy_render_workers();
    #endif
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}