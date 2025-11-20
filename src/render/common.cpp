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

color get_ray_color_simd(const ray &r, const Planet *bodies, const Trail *trails)
{
    // Ray origin and direction broadcast
    __m256 rx = _mm256_set1_ps(r.origin().x());
    __m256 ry = _mm256_set1_ps(r.origin().y());
    __m256 rz = _mm256_set1_ps(r.origin().z());

    __m256 dx = _mm256_set1_ps(r.direction().x());
    __m256 dy = _mm256_set1_ps(r.direction().y());
    __m256 dz = _mm256_set1_ps(r.direction().z());

    // a = dot(dir,dir)
    float a_scalar = dot(r.direction(), r.direction());
    __m256 a = _mm256_set1_ps(a_scalar);

    __m256 best_t = _mm256_set1_ps(1e30f);
    __m256 best_index = _mm256_set1_ps(-1.0f);

    for (int i = 0; i < NUM_BODIES; i += 8)
    {
        // Load positions
        __m256 px = _mm256_set_ps(
            bodies[i + 7].pos.x(), bodies[i + 6].pos.x(), bodies[i + 5].pos.x(),
            bodies[i + 4].pos.x(), bodies[i + 3].pos.x(), bodies[i + 2].pos.x(),
            bodies[i + 1].pos.x(), bodies[i + 0].pos.x());
        __m256 py = _mm256_set_ps(
            bodies[i + 7].pos.y(), bodies[i + 6].pos.y(), bodies[i + 5].pos.y(),
            bodies[i + 4].pos.y(), bodies[i + 3].pos.y(), bodies[i + 2].pos.y(),
            bodies[i + 1].pos.y(), bodies[i + 0].pos.y());
        __m256 pz = _mm256_set_ps(
            bodies[i + 7].pos.z(), bodies[i + 6].pos.z(), bodies[i + 5].pos.z(),
            bodies[i + 4].pos.z(), bodies[i + 3].pos.z(), bodies[i + 2].pos.z(),
            bodies[i + 1].pos.z(), bodies[i + 0].pos.z());

        // radii
        __m256 pr = _mm256_set_ps(
            bodies[i + 7].r, bodies[i + 6].r, bodies[i + 5].r,
            bodies[i + 4].r, bodies[i + 3].r, bodies[i + 2].r,
            bodies[i + 1].r, bodies[i + 0].r);

        // oc = pos - origin
        __m256 ocx = _mm256_sub_ps(px, rx);
        __m256 ocy = _mm256_sub_ps(py, ry);
        __m256 ocz = _mm256_sub_ps(pz, rz);

        // b = -2 * dot(dir, oc)
        __m256 b = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(dx, ocx),
                _mm256_add_ps(_mm256_mul_ps(dy, ocy), _mm256_mul_ps(dz, ocz))),
            _mm256_set1_ps(-2.0f));

        // c = dot(oc,oc) - r*r
        __m256 c = _mm256_sub_ps(
            _mm256_add_ps(
                _mm256_mul_ps(ocx, ocx),
                _mm256_add_ps(_mm256_mul_ps(ocy, ocy), _mm256_mul_ps(ocz, ocz))),
            _mm256_mul_ps(pr, pr));

        // discriminant = b*b - 4ac
        __m256 discriminant = _mm256_sub_ps(
            _mm256_mul_ps(b, b),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(4.0f), a), c));

        // mask for discriminant >= 0
        __m256 mask = _mm256_cmp_ps(discriminant, _mm256_set1_ps(0.0f), _CMP_GE_OQ);

        if (!_mm256_movemask_ps(mask))
            continue; // no hits in this batch

        // sqrt(discriminant)
        __m256 sqrt_d = _mm256_sqrt_ps(discriminant);

        // t = (-b - sqrt(d)) / (2a)
        __m256 denom = _mm256_mul_ps(_mm256_set1_ps(2.0f), a);
        __m256 t = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), b), sqrt_d), denom);

        // t >= 0 mask
        __m256 t_pos = _mm256_cmp_ps(t, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
        __m256 valid = _mm256_and_ps(mask, t_pos);

        if (!_mm256_movemask_ps(valid))
            continue;

        // Compare with best t so far
        __m256 is_closer = _mm256_cmp_ps(t, best_t, _CMP_LT_OQ);
        __m256 update_mask = _mm256_and_ps(valid, is_closer);

        // Update best t
        best_t = _mm256_blendv_ps(best_t, t, update_mask);

        // Update best index
        __m256 indices = _mm256_set_ps(
            (float)(i + 7), (float)(i + 6), (float)(i + 5), (float)(i + 4),
            (float)(i + 3), (float)(i + 2), (float)(i + 1), (float)(i + 0));
        best_index = _mm256_blendv_ps(best_index, indices, update_mask);
    }

    // Extract nearest hit
    alignas(32) float t_values[8];
    alignas(32) float idx_values[8];
    _mm256_store_ps(t_values, best_t);
    _mm256_store_ps(idx_values, best_index);

    float t_best = 1e30f;
    int final_idx = -1;

    for (int k = 0; k < 8; k++)
    {
        if (idx_values[k] >= 0 && t_values[k] < t_best)
        {
            t_best = t_values[k];
            final_idx = (int)idx_values[k];
        }
    }

    if (final_idx < 0)
        return {0, 0, 0, 255};

    // Now compute shading (scalar)
    vec3 hit_point = r.at(t_best);
    vec3 N = unit_vector(hit_point - bodies[final_idx].pos);
    float brightness = 0.2f + 0.8f * std::max(dot(N, light_dir), 0.0f);

    color out;
    out.r = (uint8_t)(bodies[final_idx].col.r * brightness);
    out.g = (uint8_t)(bodies[final_idx].col.g * brightness);
    out.b = (uint8_t)(bodies[final_idx].col.b * brightness);
    out.a = 255;
    return out;
}