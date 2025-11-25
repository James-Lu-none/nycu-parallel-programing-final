#include "planet.hpp"
#include "vec3.hpp"
#include "canvas.hpp"

vec3 random_vec3(float min, float max)
{
    ZoneScopedN("random_float");
    float range = (max - min);
    float div = RAND_MAX / range;
    return vec3(
        min + (rand() / div),
        min + (rand() / div),
        min + (rand() / div));
}
void load_planets_to_SoA(vector<Planet> &bodies, PlanetsSoA &bodies_soa)
{
    bodies_soa.count = bodies.size();
    bodies_soa.x = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.y = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.z = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.vx = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.vy = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.vz = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.ax = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.ay = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.az = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.mass = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.r = (float *)aligned_alloc(32, sizeof(float) * bodies.size());
    bodies_soa.col_r = (uint8_t *)malloc(sizeof(uint8_t) * bodies.size());
    bodies_soa.col_g = (uint8_t *)malloc(sizeof(uint8_t) * bodies.size());
    bodies_soa.col_b = (uint8_t *)malloc(sizeof(uint8_t) * bodies.size());
    bodies_soa.col_a = (uint8_t *)malloc(sizeof(uint8_t) * bodies.size());
    for (int i = 0; i < bodies.size(); ++i)
    {
        bodies_soa.x[i] = bodies[i].pos.x();
        bodies_soa.y[i] = bodies[i].pos.y();
        bodies_soa.z[i] = bodies[i].pos.z();
        bodies_soa.vx[i] = bodies[i].vel.x();
        bodies_soa.vy[i] = bodies[i].vel.y();
        bodies_soa.vz[i] = bodies[i].vel.z();
        bodies_soa.ax[i] = bodies[i].acc.x();
        bodies_soa.ay[i] = bodies[i].acc.y();
        bodies_soa.az[i] = bodies[i].acc.z();
        bodies_soa.mass[i] = bodies[i].mass;
        bodies_soa.r[i] = bodies[i].r;
        bodies_soa.col_r[i] = bodies[i].col.r;
        bodies_soa.col_g[i] = bodies[i].col.g;
        bodies_soa.col_b[i] = bodies[i].col.b;
        bodies_soa.col_a[i] = bodies[i].col.a;
    }
}

void load_planets_from_file(const char *filename, vector<Planet>& b)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        srand(time(NULL));
        printf("Error: Cannot open file %s\n", filename);
        printf("Generating %d random bodies instead.\n", NUM_BODIES);
        // Generate random bodies
        b.resize(NUM_BODIES);
        for (uint64_t i = 0; i < NUM_BODIES; ++i)
        {
            b[i].pos = random_vec3(-10.0f, 10.0f);
            b[i].vel = random_vec3(-10.0f, 10.0f);
            b[i].acc = vec3(0.0f, 0.0f, 0.0f);
            b[i].mass = float(rand()) / RAND_MAX * 50.0f + 10.0f;
            b[i].r = float(rand()) / RAND_MAX * 5.0f + 2.0f;
            b[i].col = color{
                uint8_t(rand() % 256),
                uint8_t(rand() % 256),
                uint8_t(rand() % 256),
                255};
        }
    }
    else
    {
        printf("Loading bodies from file %s\n", filename);
        int i = 0;

        while (true)
        {
            float x, y, z;
            float vx, vy, vz;
            float ax, ay, az;
            float mass, radius;
            int col_r, col_g, col_b, col_a;

            int n = fscanf(fp,
                           "%f %f %f %f %f %f %f %f %f %f %f %d %d %d %d",
                           &x, &y, &z,
                           &vx, &vy, &vz,
                           &ax, &ay, &az,
                           &mass,
                           &radius,
                           &col_r, &col_g, &col_b, &col_a);

            if (n == EOF || n == 0)
                break;

            if (n != 15)
            {
                fprintf(stderr, "Error: Invalid format near planet %llu (only read %d fields)\n",
                        (unsigned long long)i, n);
                exit(EXIT_FAILURE);
            }

            b.push_back(Planet{}); // Only push if we have valid data

            // Fill planet
            b[i].pos = point3(x, y, z);
            b[i].vel = vec3(vx, vy, vz);
            b[i].acc = vec3(ax, ay, az);

            b[i].mass = mass;
            b[i].r = radius;

            b[i].col = color{
                (uint8_t)col_r,
                (uint8_t)col_g,
                (uint8_t)col_b,
                (uint8_t)col_a};

            i++;
        }
        printf("Loaded %d bodies from file.\n", (int)b.size());
        fclose(fp);
    }
}

vec3 get_center_of_mass(vector<Planet>& b)
{
    ZoneScopedN("recenter");
    vec3 com_pos(0.0, 0.0, 0.0);
    float total_mass = 0.0;
    for (int i = 0; i < b.size(); ++i)
    {
        com_pos += b[i].mass * b[i].pos;
        total_mass += b[i].mass;
    }
    com_pos = com_pos / total_mass;
    return com_pos;
}

void trail_push(Trail *t, vec3 pos)
{
    ZoneScopedN("trail_push");
    if (t->size == 0)
    {
        t->pos[0] = pos;
        t->head = 1 % TRAIL_BUF;
        t->size = 1;
        return;
    }
    int last = (t->head - 1 + TRAIL_BUF) % TRAIL_BUF;
    if ((pos - t->pos[last]).length() >= MIN_DIST)
    {
        t->pos[t->head] = pos;
        t->head = (t->head + 1) % TRAIL_BUF;
        if (t->size < TRAIL_BUF)
            t->size++;
    }
}