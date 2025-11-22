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
            b.push_back(Planet{});
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
                break; // clean EOF

            if (n != 15)
            {
                fprintf(stderr, "Error: Invalid format near planet %llu (only read %d fields)\n",
                        (unsigned long long)i, n);
                exit(EXIT_FAILURE);
            }

            // Fill planet
            b[i].pos = point3(x, y, z);
            b[i].vel = vec3(vx, vy, vz);
            b[i].acc = vec3(ax, ay, az);

            b[i].mass = mass;
            b[i].r = radius; // <-- directly from file

            b[i].col = color{
                (uint8_t)col_r,
                (uint8_t)col_g,
                (uint8_t)col_b,
                (uint8_t)col_a};

            i++;
        }

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