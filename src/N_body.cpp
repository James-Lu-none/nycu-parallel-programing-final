#include "config.hpp"
#include "accelerations.hpp"
#include "planet.hpp"

#include <SDL2/SDL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tracy/Tracy.hpp"

namespace {

constexpr Uint32 kBackgroundColor = 0x00000000;

struct Trail {
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> z;
    std::size_t head = 0;
    std::size_t size = 0;

    Trail() = default;
    explicit Trail(std::size_t capacity) { resize(capacity); }

    void resize(std::size_t capacity) {
        x.assign(capacity, 0);
        y.assign(capacity, 0);
        z.assign(capacity, 0);
        head = 0;
        size = 0;
    }

    [[nodiscard]] std::size_t capacity() const { return x.size(); }
};

static double random_double(double min, double max) {
    ZoneScopedN("random_double");
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

static void fill_circle(SDL_Surface *surf, int cx, int cy, int cz, int rad, Uint32 col) {
    ZoneScopedN("fill_circle");
    if (surf == nullptr) {
        return;
    }

    const double view_z = config::VIEW_Z;
    int scaled_rad = static_cast<int>(rad * view_z / (view_z + static_cast<double>(cz)));
    int rad2 = scaled_rad * scaled_rad;
    double brightness = (view_z - static_cast<double>(cz)) / view_z;
    if (brightness < 0.2) {
        brightness = 0.2;
    }
    if (brightness > 1.0) {
        brightness = 1.0;
    }

    Uint8 r = (col >> 16) & 0xff;
    Uint8 g = (col >> 8) & 0xff;
    Uint8 b = col & 0xff;
    r = static_cast<Uint8>(r * brightness);
    g = static_cast<Uint8>(g * brightness);
    b = static_cast<Uint8>(b * brightness);
    Uint32 adj_col = (r << 16) | (g << 8) | b;

    for (int dy = -scaled_rad; dy <= scaled_rad; ++dy) {
        for (int dx = -scaled_rad; dx <= scaled_rad; ++dx) {
            if (dx * dx + dy * dy <= rad2) {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < config::WIDTH && py >= 0 && py < config::HEIGHT) {
                    SDL_Rect pixel{px, py, 1, 1};
                    SDL_FillRect(surf, &pixel, adj_col);
                }
            }
        }
    }
}

static void trail_push(Trail &t, int x, int y, int z, double min_dist) {
    ZoneScopedN("trail_push");
    if (t.capacity() == 0) {
        return;
    }

    if (t.size == 0) {
        t.x[0] = x;
        t.y[0] = y;
        t.z[0] = z;
        t.head = (t.head + 1) % t.capacity();
        t.size = 1;
        return;
    }

    std::size_t last = (t.head + t.capacity() - 1) % t.capacity();
    if (std::abs(x - t.x[last]) >= min_dist ||
        std::abs(y - t.y[last]) >= min_dist ||
        std::abs(z - t.z[last]) >= min_dist) {
        t.x[t.head] = x;
        t.y[t.head] = y;
        t.z[t.head] = z;
        t.head = (t.head + 1) % t.capacity();
        if (t.size < t.capacity()) {
            ++t.size;
        }
    }
}

static void trail_draw(SDL_Surface *surf, const Trail &t, Uint32 col) {
    ZoneScopedN("trail_draw");
    if (surf == nullptr) {
        return;
    }

    const double view_z = config::VIEW_Z;
    for (std::size_t i = 0; i < t.size; ++i) {
        std::size_t idx = (t.head + t.capacity() - 1 - i) % t.capacity();
        int scaled_rad = static_cast<int>(2 * view_z / (view_z + static_cast<double>(t.z[idx])));
        double brightness = (view_z - static_cast<double>(t.z[idx])) / view_z;
        if (brightness < 0.2) {
            brightness = 0.2;
        }
        if (brightness > 1.0) {
            brightness = 1.0;
        }
        Uint8 r = (col >> 16) & 0xff;
        Uint8 g = (col >> 8) & 0xff;
        Uint8 b = col & 0xff;
        r = static_cast<Uint8>(r * brightness);
        g = static_cast<Uint8>(g * brightness);
        b = static_cast<Uint8>(b * brightness);
        Uint32 adj_col = (r << 16) | (g << 8) | b;
        SDL_Rect p{t.x[idx], t.y[idx], scaled_rad, scaled_rad};
        SDL_FillRect(surf, &p, adj_col);
    }
}

static void step_leapfrog(Planet *bodies, std::size_t count, double dt) {
    static bool first = true;

    if (count == 0) {
        return;
    }

    if (first) {
        ZoneScopedN("step_leapfrog_first");
        accelerations(bodies, count);
        first = false;
    }

    for (std::size_t i = 0; i < count; ++i) {
        bodies[i].vx += 0.5 * bodies[i].ax * dt;
        bodies[i].vy += 0.5 * bodies[i].ay * dt;
        bodies[i].vz += 0.5 * bodies[i].az * dt;
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }

    accelerations(bodies, count);

    for (std::size_t i = 0; i < count; ++i) {
        ZoneScopedN("step_leapfrog");
        bodies[i].vx += 0.5 * bodies[i].ax * dt;
        bodies[i].vy += 0.5 * bodies[i].ay * dt;
        bodies[i].vz += 0.5 * bodies[i].az * dt;
    }
}

static void recenter(std::vector<Planet> &bodies) {
    ZoneScopedN("recenter");
    if (bodies.empty()) {
        return;
    }

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double M = 0.0;
    for (const auto &body : bodies) {
        cx += body.x * body.mass;
        cy += body.y * body.mass;
        cz += body.z * body.mass;
        M += body.mass;
    }

    cx /= M;
    cy /= M;
    cz /= M;

    double dx = static_cast<double>(config::WIDTH) / 2.0 - cx;
    double dy = static_cast<double>(config::HEIGHT) / 2.0 - cy;
    double dz = 0.0 - cz;

    for (auto &body : bodies) {
        body.x += dx;
        body.y += dy;
        body.z += dz;
    }
}

struct Vec3 {
    double x;
    double y;
    double z;
};

static Vec3 normalize(const Vec3 &v) {
    double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.0) {
        return {0.0, 0.0, 0.0};
    }
    return {v.x / len, v.y / len, v.z / len};
}

static double dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static void render_ray_traced(SDL_Surface *surf, const std::vector<Planet> &bodies,
                              const std::vector<Uint32> &colors) {
    ZoneScopedN("render_ray_traced");
    if (!surf) {
        return;
    }

    if (SDL_MUSTLOCK(surf)) {
        SDL_LockSurface(surf);
    }

    Uint32 *pixels = static_cast<Uint32 *>(surf->pixels);
    const int pitch = surf->pitch / static_cast<int>(sizeof(Uint32));

    const Vec3 camera{static_cast<double>(config::WIDTH) / 2.0,
                      static_cast<double>(config::HEIGHT) / 2.0,
                      -config::VIEW_Z};
    const Vec3 light_dir = normalize({-0.3, -0.6, -1.0});

    SDL_PixelFormat *fmt = surf->format;

    for (int py = 0; py < config::HEIGHT; ++py) {
        for (int px = 0; px < config::WIDTH; ++px) {
            Vec3 pixel_pos{static_cast<double>(px), static_cast<double>(py), 0.0};
            Vec3 dir{pixel_pos.x - camera.x, pixel_pos.y - camera.y, -camera.z};
            dir = normalize(dir);

            double closest_t = std::numeric_limits<double>::infinity();
            int closest_idx = -1;

            for (std::size_t i = 0; i < bodies.size(); ++i) {
                const Planet &body = bodies[i];
                Vec3 oc{camera.x - body.x, camera.y - body.y, camera.z - body.z};
                double b = 2.0 * dot(oc, dir);
                double c = dot(oc, oc) - body.r * body.r;
                double discriminant = b * b - 4.0 * c;
                if (discriminant < 0.0) {
                    continue;
                }
                double sqrt_disc = std::sqrt(discriminant);
                double t0 = (-b - sqrt_disc) / 2.0;
                double t1 = (-b + sqrt_disc) / 2.0;

                double t = t0;
                if (t < 0.0) {
                    t = t1;
                }
                if (t > 0.0 && t < closest_t) {
                    closest_t = t;
                    closest_idx = static_cast<int>(i);
                }
            }

            Uint32 final_color = SDL_MapRGB(fmt, 0, 0, 0);

            if (closest_idx >= 0) {
                const Planet &body = bodies[static_cast<std::size_t>(closest_idx)];
                Vec3 hit_point{camera.x + dir.x * closest_t,
                               camera.y + dir.y * closest_t,
                               camera.z + dir.z * closest_t};
                Vec3 normal{(hit_point.x - body.x) / body.r,
                            (hit_point.y - body.y) / body.r,
                            (hit_point.z - body.z) / body.r};
                normal = normalize(normal);
                double diffuse = std::max(0.0, -dot(normal, light_dir));
                double ambient = 0.2;
                double intensity = std::clamp(ambient + 0.8 * diffuse, 0.0, 1.0);

                Uint8 r, g, b;
                SDL_GetRGB(colors[closest_idx % colors.size()], fmt, &r, &g, &b);
                r = static_cast<Uint8>(std::min(255.0, r * intensity));
                g = static_cast<Uint8>(std::min(255.0, g * intensity));
                b = static_cast<Uint8>(std::min(255.0, b * intensity));
                final_color = SDL_MapRGB(fmt, r, g, b);
            }

            pixels[py * pitch + px] = final_color;
        }
    }

    if (SDL_MUSTLOCK(surf)) {
        SDL_UnlockSurface(surf);
    }
}

static void render_rasterized(SDL_Surface *surf, const std::vector<Planet> &bodies,
                              const std::vector<Trail> &trails,
                              const std::vector<Uint32> &colors) {
    ZoneScopedN("render_rasterized");
    SDL_FillRect(surf, nullptr, kBackgroundColor);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        trail_draw(surf, trails[i], colors[i % colors.size()]);
        fill_circle(surf,
                    static_cast<int>(bodies[i].x),
                    static_cast<int>(bodies[i].y),
                    static_cast<int>(bodies[i].z),
                    static_cast<int>(bodies[i].r),
                    colors[i % colors.size()]);
    }
}

} // namespace

namespace config {

RuntimeOptions parse_arguments(int argc, char **argv) {
    RuntimeOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "-n" || arg == "--bodies") && i + 1 < argc) {
            options.body_count = std::stoull(argv[++i]);
        } else if ((arg == "-t" || arg == "--trail") && i + 1 < argc) {
            options.trail_length = std::stoull(argv[++i]);
        } else if (arg == "--raytraced") {
            options.start_in_ray_tracing_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: N_body [options]\n"
                      << "  -n, --bodies <count>   Number of simulated bodies (default "
                      << DEFAULT_BODY_COUNT << ")\n"
                      << "  -t, --trail <length>   Trail history length (default "
                      << DEFAULT_TRAIL_LENGTH << ")\n"
                      << "      --raytraced        Start in ray-traced rendering mode\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    if (options.body_count == 0) {
        throw std::invalid_argument("Body count must be greater than zero.");
    }
    if (options.trail_length == 0) {
        options.trail_length = 1;
    }
    return options;
}

} // namespace config

int main(int argc, char **argv) {
    tracy::SetThreadName("main_thread");
    srand(static_cast<unsigned int>(time(nullptr)));

    config::RuntimeOptions options;
    try {
        options = config::parse_arguments(argc, argv);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("N-body Simulation",
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       config::WIDTH,
                                       config::HEIGHT,
                                       SDL_WINDOW_SHOWN);
    if (!win) {
        std::cerr << "SDL_CreateWindow: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    SDL_Surface *surf = SDL_GetWindowSurface(win);
    if (!surf) {
        std::cerr << "SDL_GetWindowSurface: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    const std::vector<Uint32> colors = {
        0x00ff0000, 0x0000ff00, 0x000000ff,
        0x00ffff00, 0x00ff00ff, 0x0000ffff};

    std::vector<Planet> bodies(options.body_count);
    const double S = 140.0;
    const double VS = 140.0;
    const double m = 200.0;
    double cx = static_cast<double>(config::WIDTH) / 2.0;
    double cy = static_cast<double>(config::HEIGHT) / 2.0;
    double cz = 0.0;

    for (auto &body : bodies) {
        body = Planet{
            cx + random_double(-1.0, 1.0) * S,
            cy + random_double(-1.0, 1.0) * S,
            cz + random_double(-0.1, 0.1) * S,
            random_double(-1.0, 1.0) * VS,
            random_double(-1.0, 1.0) * VS,
            random_double(-0.1, 0.1) * VS,
            0.0, 0.0, 0.0,
            m, 15.0};
    }

    std::vector<Trail> trails;
    trails.reserve(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        trails.emplace_back(options.trail_length);
    }

    accelerations_setup(bodies.data(), bodies.size());

    bool running = true;
    bool ray_tracing_enabled = options.start_in_ray_tracing_mode;
    SDL_Event ev;
    constexpr double FIXED_DT = 0.0002;
    double accumulator = 0.0;
    Uint32 prev = SDL_GetTicks();

    while (running) {
        FrameMark;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_r) {
                ray_tracing_enabled = !ray_tracing_enabled;
            }
        }

        Uint32 now = SDL_GetTicks();
        double frame_dt = (now - prev) / 1000.0;
        prev = now;
        if (frame_dt > 0.05) {
            frame_dt = 0.05;
        }
        accumulator += frame_dt;

        while (accumulator >= FIXED_DT) {
            step_leapfrog(bodies.data(), bodies.size(), FIXED_DT);
            accumulator -= FIXED_DT;
        }

        recenter(bodies);

        for (std::size_t i = 0; i < bodies.size(); ++i) {
            trail_push(trails[i], static_cast<int>(bodies[i].x),
                       static_cast<int>(bodies[i].y), static_cast<int>(bodies[i].z),
                       config::MIN_DIST);
        }

        if (ray_tracing_enabled) {
            render_ray_traced(surf, bodies, colors);
        } else {
            render_rasterized(surf, bodies, trails, colors);
        }

        SDL_UpdateWindowSurface(win);
        SDL_Delay(16);
    }

    accelerations_teardown();

    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

