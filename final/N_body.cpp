#include <cstdio>
#include <cmath>
#include <SDL2/SDL.h>
#include <pthread.h>
#include <ctime>
#include <cstdlib>

#include "tracy/public/TracyClient.cpp"
#include "tracy/public/tracy/TracyOpenCL.hpp"

using namespace std;

#define WIDTH   1200
#define HEIGHT  800

#define NUM_BODIES 3
#define TRAIL_BUF  100
#define MIN_DIST   1.5

#define G        10000.0
#define EPSILON  1e-6

#define COL_BLACK      0x00000000
    uint32_t COLORS[] = {0x00ff0000, 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff};

#define NUM_THREADS 4

#define view_z 100.0
typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double r;
} Planet;

typedef struct {
    int x[TRAIL_BUF];
    int y[TRAIL_BUF];
    int z[TRAIL_BUF];
    int head;
    int size;
} Trail;

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

typedef struct
{
    const Planet *b;
    int t_id;
    int t_N;
    double *t_ax;
    double *t_ay;
    double *t_az;
} AccelerationArgsV1;

static void *accelerations_thread_v1(void *arg)
{
    AccelerationArgsV1 *A = (AccelerationArgsV1 *)arg;
    const Planet *b = A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;

    double *ax = A->t_ax;
    double *ay = A->t_ay;
    double *az = A->t_az;

    // add t_N-1 to NUM_BODIES before divide by t_N to ensure all acceleration pairs (i,j) are covered
    int chunk = (NUM_BODIES + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < NUM_BODIES) ? (i_start + chunk) : NUM_BODIES;
    int count = 0;

    for (int i = i_start; i < i_end; ++i)
    {
        for (int j = i + 1; j < NUM_BODIES; ++j)
        {
            double dx = b[j].x - b[i].x;
            double dy = b[j].y - b[i].y;
            double dz = b[j].z - b[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
            double dist = sqrt(dist2);

            double F = (G * b[i].mass * b[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;

            ax[i] += fx / b[i].mass;
            ay[i] += fy / b[i].mass;
            az[i] += fz / b[i].mass;
            ax[j] -= fx / b[j].mass;
            ay[j] -= fy / b[j].mass;
            az[j] -= fz / b[j].mass;
            count++;
        }
    }
    printf("Thread %d: i_start=%d, i_end=%d, calc_count=%d\n", t_id, i_start, i_end, count);
    return NULL;
}

static void *accelerations_thread_v2(void *arg)
{
    AccelerationArgsV1 *A = (AccelerationArgsV1 *)arg;
    const Planet *b = A->b;
    int t_id = A->t_id;
    int t_N = A->t_N;

    double *ax = A->t_ax;
    double *ay = A->t_ay;
    double *az = A->t_az;

    // add t_N-1 to NUM_BODIES before divide by t_N to ensure all acceleration pairs (i,j) are covered
    int chunk = (NUM_BODIES + t_N - 1) / t_N;
    int i_start = t_id * chunk;
    int i_end = (i_start + chunk < NUM_BODIES) ? (i_start + chunk) : NUM_BODIES;
    int count = 0;

    for (int i = i_start; i < i_end; ++i)
    {
        for (int j = 0; j < NUM_BODIES; ++j)
        {
            double dx = b[j].x - b[i].x;
            double dy = b[j].y - b[i].y;
            double dz = b[j].z - b[i].z;
            double dist2 = dx * dx + dy * dy + dz * dz + EPSILON;
            double dist = sqrt(dist2);

            double F = (G * b[i].mass * b[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;
            ax[i] += fx / b[i].mass;
            ay[i] += fy / b[i].mass;
            az[i] += fz / b[i].mass;
            count++;
        }
    }
    printf("Thread %d: i_start=%d, i_end=%d, calc_count=%d\n", t_id, i_start, i_end, count);
    return NULL;
}

static void accelerations_parallel(Planet b[])
{
    ZoneScopedN("accelerations_parallel");

    int t_N = NUM_THREADS > NUM_BODIES ? NUM_BODIES : NUM_THREADS;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * t_N);
    AccelerationArgsV1 *args = (AccelerationArgsV1 *)malloc(sizeof(AccelerationArgsV1) * t_N);

    double **t_ax = (double **)malloc(sizeof(double *) * t_N);
    double **t_ay = (double **)malloc(sizeof(double *) * t_N);
    double **t_az = (double **)malloc(sizeof(double *) * t_N);
    for (int t = 0; t < t_N; ++t)
    {
        t_ax[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_ay[t] = (double *)calloc(NUM_BODIES, sizeof(double));
        t_az[t] = (double *)calloc(NUM_BODIES, sizeof(double));
    }

    for (int i = 0; i < NUM_BODIES; ++i) b[i].ax = b[i].ay = b[i].az = 0.0;

    for (int t = 0; t < t_N; ++t)
    {
        args[t].b = b;
        args[t].t_id = t;
        args[t].t_N = t_N;
        args[t].t_ax = t_ax[t];
        args[t].t_ay = t_ay[t];
        args[t].t_az = t_az[t];
        pthread_create(&threads[t], NULL, accelerations_thread_v2, &args[t]);
    }

    for (int t = 0; t < t_N; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < t_N; ++t)
    {
        for (int i = 0; i < NUM_BODIES; ++i)
        {
            b[i].ax += t_ax[t][i];
            b[i].ay += t_ay[t][i];
            b[i].az += t_az[t][i];
        }
    }
    
    for (int t = 0; t < t_N; ++t)
    {
        free(t_ax[t]);
        free(t_ay[t]);
        free(t_az[t]);
    }
    free(t_ax);
    free(t_ay);
    free(t_az);
    free(threads);
    free(args);
}

static void accelerations(const Planet b[], double ax[], double ay[], double az[])
{
    ZoneScopedN("accelerations");
    for (int i = 0; i < NUM_BODIES; ++i) ax[i] = ay[i] = az[i] = 0.0;

    for (int i = 0; i < NUM_BODIES; ++i) {
        for (int j = i + 1; j < NUM_BODIES; ++j) {
            double dx = b[j].x - b[i].x;
            double dy = b[j].y - b[i].y;
            double dz = b[j].z - b[i].z;
            double dist2 = dx*dx + dy*dy + dz*dz + EPSILON;
            double dist  = sqrt(dist2);

            double F  = (G * b[i].mass * b[j].mass) / dist2;
            double fx = F * dx / dist;
            double fy = F * dy / dist;
            double fz = F * dz / dist;

            ax[i] +=  fx / b[i].mass;
            ay[i] +=  fy / b[i].mass;
            az[i] +=  fz / b[i].mass;
            ax[j] -=  fx / b[j].mass;
            ay[j] -=  fy / b[j].mass;
            az[j] -=  fz / b[j].mass;
        }
    }
}

static void step_leapfrog(Planet b[], double dt)
{
    static int first = 1;

    if (first) {
        ZoneScopedN("step_leapfrog_first");
        accelerations_parallel(b);
        first = 0;
    }

    for (int i = 0; i < NUM_BODIES; ++i) {
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
        b[i].x  +=      b[i].vx * dt;
        b[i].y  +=      b[i].vy * dt;
        b[i].z  +=      b[i].vz * dt;
    }

    accelerations_parallel(b);

    for (int i = 0; i < NUM_BODIES; ++i) {
        ZoneScopedN("step_leapfrog");
        b[i].vx += 0.5 * b[i].ax * dt;
        b[i].vy += 0.5 * b[i].ay * dt;
        b[i].vz += 0.5 * b[i].az * dt;
    }
}

static void recenter(Planet b[])
{
    ZoneScopedN("recenter");
    double cx = 0, cy = 0, cz = 0, M = 0;
    for (int i = 0; i < NUM_BODIES; ++i) {
        cx += b[i].x * b[i].mass;
        cy += b[i].y * b[i].mass;
        cz += b[i].z * b[i].mass;
        M  += b[i].mass;
    }
    cx /= M;
    cy /= M;
    cz /= M;

    double dx = WIDTH / 2.0 - cx;
    double dy = HEIGHT / 2.0 - cy;
    double dz = 0.0 - cz;
    for (int i = 0; i < NUM_BODIES; ++i) {
        b[i].x += dx;
        b[i].y += dy;
        b[i].z += dz;
    }
}

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
    
    while (running) {
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

        for (int i = 0; i < NUM_BODIES; ++i)
            trail_push(&trails[i], (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z);

        SDL_FillRect(surf, NULL, COL_BLACK);
        
        for (int i = 0; i < NUM_BODIES; ++i) {
            trail_draw(surf, &trails[i], COLORS[i % 6]);
            fill_circle(surf, (int)bodies[i].x, (int)bodies[i].y, (int)bodies[i].z, (int) bodies[i].r, COLORS[i % 6]);
        }

        SDL_UpdateWindowSurface(win);
        SDL_Delay(16);
    }

    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}