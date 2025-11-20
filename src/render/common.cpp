#include "render.hpp"
#include "planet.hpp"
#include "vec3.hpp"
#include "ray.hpp"

const vec3 light_dir = unit_vector(vec3(1, 1, 0.6));

float hit_trail(const Trail &t, const ray &r)
{
    int radius_squared = 4; // radius 2
    for (int i = 0; i < t.size; ++i)
    {
        vec3 oc = t.pos[i] - r.origin();
        float a = dot(r.direction(), r.direction());
        float b = -2.0 * dot(r.direction(), oc);
        float c = dot(oc, oc) - radius_squared;
        float discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            return true;
        }
    }
    return false;
}

float hit_planet(const Planet &p, const ray &r)
{
    vec3 oc = p.pos - r.origin();
    float a = dot(r.direction(), r.direction());
    float b = -2.0 * dot(r.direction(), oc);
    float c = dot(oc, oc) - p.r * p.r;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

color background(const ray &r)
{
    color hit_color = {0, 0, 0, 255};

    #ifdef DEBUG
    const vec3 O = r.origin();
    const vec3 D = r.direction();
    constexpr float radius2 = 5.0f;
    float min_t = 1e30;
    // X-axis cylinder: y^2 + z^2 = r^2
    printf("Ray origin: (%.2f, %.2f, %.2f), direction: (%.2f, %.2f, %.2f)\n",
        O.x(), O.y(), O.z(), D.x(), D.y(), D.z());
    {
        float a = D.y() * D.y() + D.z() * D.z();
        float b = 2.0f * (O.y() * D.y() + O.z() * D.z());
        float c = O.y() * O.y() + O.z() * O.z() - radius2;
        
        float disc = b * b - 4 * a * c;
        float t = (-b - sqrt(disc)) / (2.0 * a);
        if (disc >= 0.0f && t > 0.0f && t < min_t)
        {
            min_t = t;
            uint8_t brightness = r.at(t).x() > 0 ? 255 : 70;
            hit_color = {brightness, 0, 0, 255};
        }
    }

    // Y-axis cylinder: x^2 + z^2 = r^2
    {
        float a = D.x() * D.x() + D.z() * D.z();
        float b = 2.0f * (O.x() * D.x() + O.z() * D.z());
        float c = O.x() * O.x() + O.z() * O.z() - radius2;

        float disc = b * b - 4 * a * c;
        float t = (-b - sqrt(disc)) / (2.0 * a);
        if (disc >= 0.0f && t > 0.0f && t < min_t)
        {
            min_t = t;
            uint8_t brightness = r.at(t).y() > 0 ? 255 : 70;
            hit_color = {0, brightness, 0, 255};
        }
    }
    // Z-axis cylinder: x^2 + y^2 = r^2
    {
        float a = D.x() * D.x() + D.y() * D.y();
        float b = 2.0f * (O.x() * D.x() + O.y() * D.y());
        float c = O.x() * O.x() + O.y() * O.y() - radius2;

        float disc = b * b - 4 * a * c;
        float t = (-b - sqrt(disc)) / (2.0f * a);
        if (disc >= 0.0f && t > 0.0f && t < min_t)
        {
            min_t = t;
            uint8_t brightness = r.at(t).z() > 0 ? 255 : 70;
            hit_color = {0, 0, brightness, 255};
        }
    }
    // center: x^2 + y^2 + z^2 = r^2
    {
        float a = D.x() * D.x() + D.y() * D.y() + D.z() * D.z();
        float b = 2.0f * (O.x() * D.x() + O.y() * D.y() + O.z() * D.z());
        float c = O.x() * O.x() + O.y() * O.y() + O.z() * O.z() - radius2*radius2;

        float disc = b * b - 4 * a * c;
        float t = (-b - sqrt(disc)) / (2.0 * a);
        if (disc >= 0.0f && t > 0.0f && t < min_t)
        {
            min_t = (-b - sqrt(disc)) / (2.0 * a);
            hit_color = {255, 255, 255, 255};
        }
    }
    #endif //DEBUG

    return hit_color;
}

color get_ray_color(const ray &r, const Planet *bodies, const Trail *trails)
{
    float t = 1e30;
    int idx = -1;
    for (int i = 0; i < NUM_BODIES; ++i)
    {
        float t_i = hit_planet(bodies[i], r);
        if (t_i >= 0 && t_i < t) {
            t = t_i;
            idx = i;
        }
    }
    if (t >= 0 && idx != -1)
    {
        vec3 hit_point = r.at(t);
        vec3 N = unit_vector(hit_point - bodies[idx].pos);
        // calculate brightness with dot product of normal and light direction
        float brightness = 0.2f + 0.8f * std::max(dot(N, light_dir), 0.0f);

        return {
            (uint8_t)(bodies[idx].col.r * brightness),
            (uint8_t)(bodies[idx].col.g * brightness),
            (uint8_t)(bodies[idx].col.b * brightness),
            255
        };
    }

    return background(r);
}


color get_ray_color_simd(const ray &r, const Planet* bodies, const Trail* trails)
{
    // === Ray data (float) ===
    const vec3  O  = r.origin();
    const vec3  D  = r.direction();

    float Ox = O.x();
    float Oy = O.y();
    float Oz = O.z();
    float Dx = D.x();
    float Dy = D.y();
    float Dz = D.z();

    float a_scalar = Dx*Dx + Dy*Dy + Dz*Dz;

    // AVX 常數
    __m256 a_v    = _mm256_set1_ps(a_scalar);
    __m256 two_v  = _mm256_set1_ps(2.0f);
    __m256 four_v = _mm256_set1_ps(4.0f);
    __m256 Ox_v   = _mm256_set1_ps(Ox);
    __m256 Oy_v   = _mm256_set1_ps(Oy);
    __m256 Oz_v   = _mm256_set1_ps(Oz);
    __m256 Dx_v   = _mm256_set1_ps(Dx);
    __m256 Dy_v   = _mm256_set1_ps(Dy);
    __m256 Dz_v   = _mm256_set1_ps(Dz);

    // === gather 用的 base pointer ===
    int stride_f = sizeof(Planet) / sizeof(float);

    float* base_pos = (float*)&bodies[0].pos.e[0];
    float* base_r =
        (float*)((char*)&bodies[0] + offsetof(Planet, r));

    int i = 0;
    int vec_end = (NUM_BODIES / 8) * 8;   // AVX: 一次處理 8 顆 planet

    for (; i < vec_end; i += 8)
    {
        // 8 個 index（以 float index 計算）
        int idx_arr[8] = {
            (i+0)*stride_f,
            (i+1)*stride_f,
            (i+2)*stride_f,
            (i+3)*stride_f,
            (i+4)*stride_f,
            (i+5)*stride_f,
            (i+6)*stride_f,
            (i+7)*stride_f
        };

        __m256i vidx = _mm256_loadu_si256((__m256i*)idx_arr);

        // === gather pos.x, pos.y, pos.z ===
        __m256 Px = _mm256_i32gather_ps(base_pos + 0, vidx, 4);
        __m256 Py = _mm256_i32gather_ps(base_pos + 1, vidx, 4);
        __m256 Pz = _mm256_i32gather_ps(base_pos + 2, vidx, 4);

        // === gather radius ===
        __m256 R = _mm256_i32gather_ps(base_r, vidx, 4);

        // oc = p.pos - O
        __m256 ocx = _mm256_sub_ps(Px, Ox_v);
        __m256 ocy = _mm256_sub_ps(Py, Oy_v);
        __m256 ocz = _mm256_sub_ps(Pz, Oz_v);

        // dot(D, oc)
        __m256 dot_doc =
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(Dx_v, ocx),
                    _mm256_mul_ps(Dy_v, ocy)
                ),
                _mm256_mul_ps(Dz_v, ocz)
            );

        // b = -2 * dot(D, oc)
        __m256 b_v = _mm256_mul_ps(_mm256_set1_ps(-2.0f), dot_doc);

        // |oc|^2
        __m256 oc2 =
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(ocx, ocx),
                    _mm256_mul_ps(ocy, ocy)
                ),
                _mm256_mul_ps(ocz, ocz)
            );

        // r^2
        __m256 R2 = _mm256_mul_ps(R, R);

        // c = |oc|^2 - r^2
        __m256 c_v = _mm256_sub_ps(oc2, R2);

        // disc = b*b - 4ac
        __m256 disc =
            _mm256_sub_ps(
                _mm256_mul_ps(b_v, b_v),
                _mm256_mul_ps(four_v, _mm256_mul_ps(a_v, c_v))
            );

        // === scalar 檢查 8 lanes ===
        float disc_arr[8], b_arr[8];
        _mm256_storeu_ps(disc_arr, disc);
        _mm256_storeu_ps(b_arr,    b_v);

        for (int k = 0; k < 8; ++k)
        {
            int idx = i + k;
            if (idx >= NUM_BODIES) break;

            if (disc_arr[k] >= 0.0f)
            {
                float t = (-b_arr[k] - std::sqrt(disc_arr[k])) / (2.0f * a_scalar);
                if (t >= 0.0f)
                {
                    vec3 hit_point = r.at(t);
                    vec3 N = 128.0f * (unit_vector(hit_point - bodies[idx].pos) + vec3(1,1,1));

                    return {
                        (uint8_t)std::min(N.x(), 255.0f),
                        (uint8_t)std::min(N.y(), 255.0f),
                        (uint8_t)std::min(N.z(), 255.0f),
                        255
                    };
                }
            }
        }
        for (int k = 0; k < 8; ++k)
        {
            int idx = i + k;
            if (idx >= NUM_BODIES) break;

            if (disc_arr[k] >= 0.0f)
            {
                float t = (-b_arr[k] - std::sqrt(disc_arr[k])) / (2.0f * a_scalar);
                if (t >= 0.0f)
                {
                    vec3 hit_point = r.at(t);
                    vec3 N = 128.0f * (unit_vector(hit_point - bodies[idx].pos) + vec3(1,1,1));

                    return {
                        (uint8_t)std::min(N.x(), 255.0f),
                        (uint8_t)std::min(N.y(), 255.0f),
                        (uint8_t)std::min(N.z(), 255.0f),
                        255
                    };
                }
            }
        }
    }
    return {0,0,0,255};
}