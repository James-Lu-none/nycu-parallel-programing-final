#include "config.hpp"
#include "accelerations.hpp"
#include <vector>

using namespace std;

typedef struct {
    int x[TRAIL_BUF];
    int y[TRAIL_BUF];
    int z[TRAIL_BUF];
    int head;
    int size;
} Trail;

static Uint32 hsv_to_rgb(double h, double s, double v)
{
    double c = v * s;
    double h_prime = h * 6.0;
    double x = c * (1.0 - fabs(fmod(h_prime, 2.0) - 1.0));
    double m = v - c;
    double r = 0.0, g = 0.0, b = 0.0;

    if (h_prime >= 0.0 && h_prime < 1.0) {
        r = c;
        g = x;
    } else if (h_prime < 2.0) {
        r = x;
        g = c;
    } else if (h_prime < 3.0) {
        g = c;
        b = x;
    } else if (h_prime < 4.0) {
        g = x;
        b = c;
    } else if (h_prime < 5.0) {
        r = x;
        b = c;
    } else {
        r = c;
        b = x;
    }

    Uint8 R = static_cast<Uint8>((r + m) * 255.0);
    Uint8 G = static_cast<Uint8>((g + m) * 255.0);
    Uint8 B = static_cast<Uint8>((b + m) * 255.0);
    return (R << 16) | (G << 8) | B;
}

static Uint32 color_for_index(size_t idx)
{
    static const Uint32 base_colors[] = {
        0x00ff0000, 0x0000ff00, 0x000000ff,
        0x00ffff00, 0x00ff00ff, 0x0000ffff
    };
    constexpr size_t base_count = sizeof(base_colors) / sizeof(base_colors[0]);
    if (idx < base_count) {
        return base_colors[idx];
    }

    double hue = fmod((idx - base_count) * 0.61803398875, 1.0);
    return hsv_to_rgb(hue, 0.65, 0.95);
}

static void fill_circle(SDL_Surface *surf, int cx, int cy, int cz, int rad, Uint32 col)
{
    ZoneScopedN("fill_circle");
    int scaled_rad = rad * view_z / (view_z + cz);
    int rad2 = scaled_rad * scaled_rad;
    double brightness = (view_z - cz) / view_z;
    if (brightness < 0.2)
        brightness = 0.2;
    if (brightness > 1.0)
        brightness = 1.0;
    Uint8 r = (col >> 16) & 0xff;
    Uint8 g = (col >> 8) & 0xff;
    Uint8 b = col & 0xff;
    r = (Uint8)(r * brightness);
    g = (Uint8)(g * brightness);
    b = (Uint8)(b * brightness);
    Uint32 adj_col = (r << 16) | (g << 8) | b;
    for (int dy = -scaled_rad; dy <= scaled_rad; ++dy)
    {
        for (int dx = -scaled_rad; dx <= scaled_rad; ++dx)
        {
            if (dx*dx + dy*dy <= rad2) {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    SDL_Rect pixel = { px, py, 1, 1 };
                    SDL_FillRect(surf, &pixel, adj_col);
                }
            }
        }
    }
}

static void trail_push(Trail *t, int x, int y, int z)
{
    ZoneScopedN("trail_push");
    if (t->size == 0) {
        t->x[0] = x;
        t->y[0] = y;
        t->z[0] = z;
        t->head = 1 % TRAIL_BUF;
        t->size = 1;
        return;
    }

    int last = (t->head - 1 + TRAIL_BUF) % TRAIL_BUF;
    if (abs(x - t->x[last]) >= MIN_DIST || abs(y - t->y[last]) >= MIN_DIST || abs(z - t->z[last]) >= MIN_DIST) {
        t->x[t->head] = x;
        t->y[t->head] = y;
        t->z[t->head] = z;
        t->head = (t->head + 1) % TRAIL_BUF;
        if (t->size < TRAIL_BUF) t->size++;
    }
}

static void trail_draw(SDL_Surface *surf, const Trail *t, Uint32 col)
{
    ZoneScopedN("trail_draw");
    for (int i = 0; i < t->size; ++i) {
        int idx = (t->head - 1 - i + TRAIL_BUF) % TRAIL_BUF;
        int scaled_rad = 2 * view_z / (view_z + t->z[idx]);
        double brightness = (view_z - t->z[idx]) / view_z;
        if (brightness < 0.2)
            brightness = 0.2;
        if (brightness > 1.0)
            brightness = 1.0;
        Uint8 r = (col >> 16) & 0xff;
        Uint8 g = (col >> 8) & 0xff;
        Uint8 b = col & 0xff;
        r = (Uint8)(r * brightness);
        g = (Uint8)(g * brightness);
        b = (Uint8)(b * brightness);
        Uint32 adj_col = (r << 16) | (g << 8) | b;
        SDL_Rect p = { t->x[idx], t->y[idx], scaled_rad, scaled_rad };
        SDL_FillRect(surf, &p, adj_col);
    }
}

static void step_leapfrog(vector<Planet> &b, double dt)
{
    static bool first = true;
    static size_t last_body_count = 0;
    const size_t body_count = b.size();

    if (body_count == 0) {
        return;
    }

    if (body_count != last_body_count) {
        first = true;
        last_body_count = body_count;
    }

    if (first) {
        ZoneScopedN("step_leapfrog_first");
        accelerations(b.data(), static_cast<int>(body_count));
        first = false;
    }

    for (size_t i = 0; i < body_count; ++i) {
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
        b[i].x  +=      b[i].vx * dt;
        b[i].y  +=      b[i].vy * dt;
        b[i].z  +=      b[i].vz * dt;
    }

    accelerations(b.data(), static_cast<int>(body_count));

    for (size_t i = 0; i < body_count; ++i) {
        ZoneScopedN("step_leapfrog");
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
    }
}

static void recenter(vector<Planet> &b)
{
    ZoneScopedN("recenter");
    double cx = 0, cy = 0, cz = 0, M = 0;
    for (const Planet &planet : b) {
        cx += planet.x * planet.mass;
        cy += planet.y * planet.mass;
        cz += planet.z * planet.mass;
        M  += planet.mass;
    }
    if (M == 0.0) {
        return;
    }
    cx /= M;
    cy /= M;
    cz /= M;

    double dx = WIDTH / 2.0 - cx;
    double dy = HEIGHT / 2.0 - cy;
    double dz = 0.0 - cz;
    for (Planet &planet : b) {
        planet.x += dx;
        planet.y += dy;
        planet.z += dz;
    }
}

double random_double(double min, double max)
{
    ZoneScopedN("random_double");
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int main(int argc, char *argv[])
{
    tracy::SetThreadName("main_thread");
    srand(time(NULL));

    int body_count = DEFAULT_NUM_BODIES;
    if (argc > 1) {
        char *end = nullptr;
        long parsed = strtol(argv[1], &end, 10);
        if (end && *end == '\0' && parsed > 0) {
            body_count = static_cast<int>(parsed);
        } else {
            fprintf(stderr, "Invalid body count '%s', using default %d.\n", argv[1], DEFAULT_NUM_BODIES);
        }
    }

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

    vector<Planet> bodies(static_cast<size_t>(body_count));
    const double S  = 140.0;
    const double VS = 140.0;
    const double m  = 200.0;
    double cx = WIDTH / 2.0;
    double cy = HEIGHT / 2.0;
    double cz = 0.0;

    for (Planet &planet : bodies){
        planet = (Planet){
            cx + random_double(-1.0, 1.0) * S,
            cy + random_double(-1.0, 1.0) * S,
            cz + random_double(-0.1, 0.1) * S,
            random_double(-1.0, 1.0) * VS,
            random_double(-1.0, 1.0) * VS,
            random_double(-0.1, 0.1) * VS,
            0.0, 0.0, 0.0,
            m, 15};
    }

    vector<Trail> trails(static_cast<size_t>(body_count));

    int running = 1;
    SDL_Event ev;
    const double FIXED_DT = 0.0002;
    double accumulator = 0.0;
    Uint32 prev = SDL_GetTicks();
    #ifdef INIT_REQUIRED
        init_workers(body_count);
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
            step_leapfrog(bodies, FIXED_DT);
            accumulator -= FIXED_DT;
        }

        recenter(bodies);

        for (size_t i = 0; i < bodies.size(); ++i)
            trail_push(&trails[i], (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z);

        SDL_FillRect(surf, NULL, COL_BLACK);

        for (size_t i = 0; i < bodies.size(); ++i) {
            Uint32 color = color_for_index(i);
            trail_draw(surf, &trails[i], color);
            fill_circle(surf, (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z, (int) bodies[i].r, color);
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