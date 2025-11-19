#include "render.hpp"
#include "planet.hpp"

void recenter(Planet b[])
{
    ZoneScopedN("recenter");
    double cx = 0, cy = 0, cz = 0, M = 0;
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        cx += b[i].x * b[i].mass;
        cy += b[i].y * b[i].mass;
        cz += b[i].z * b[i].mass;
        M += b[i].mass;
    }
    cx /= M;
    cy /= M;
    cz /= M;

    double dx = WIDTH / 2.0 - cx;
    double dy = HEIGHT / 2.0 - cy;
    double dz = 0.0 - cz;
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        b[i].x += dx;
        b[i].y += dy;
        b[i].z += dz;
    }
}

void fill_circle(SDL_Surface *surf, int cx, int cy, int cz, int rad, Uint32 col)
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
            if (dx * dx + dy * dy <= rad2)
            {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT)
                {
                    SDL_Rect pixel = {px, py, 1, 1};
                    SDL_FillRect(surf, &pixel, adj_col);
                }
            }
        }
    }
}
void trail_push(Trail *t, int x, int y, int z)
{
    ZoneScopedN("trail_push");
    if (t->size == 0)
    {
        t->x[0] = x;
        t->y[0] = y;
        t->z[0] = z;
        t->head = 1 % TRAIL_BUF;
        t->size = 1;
        return;
    }
    int last = (t->head - 1 + TRAIL_BUF) % TRAIL_BUF;
    if (abs(x - t->x[last]) >= MIN_DIST || abs(y - t->y[last]) >= MIN_DIST || abs(z - t->z[last]) >= MIN_DIST)
    {
        t->x[t->head] = x;
        t->y[t->head] = y;
        t->z[t->head] = z;
        t->head = (t->head + 1) % TRAIL_BUF;
        if (t->size < TRAIL_BUF)
            t->size++;
    }
}
void trail_draw(SDL_Surface *surf, const Trail *t, Uint32 col)
{
    ZoneScopedN("trail_draw");
    for (int i = 0; i < t->size; ++i)
    {
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
        SDL_Rect p = {t->x[idx], t->y[idx], scaled_rad, scaled_rad};
        SDL_FillRect(surf, &p, adj_col);
    }
}