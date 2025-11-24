#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class vec3
{
public:
    float e[3];

    CUDA_CALLABLE vec3() : e{0, 0, 0} {}
    CUDA_CALLABLE vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    CUDA_CALLABLE float x() const { return e[0]; }
    CUDA_CALLABLE float y() const { return e[1]; }
    CUDA_CALLABLE float z() const { return e[2]; }

    CUDA_CALLABLE vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    CUDA_CALLABLE float operator[](int i) const { return e[i]; }
    CUDA_CALLABLE float &operator[](int i) { return e[i]; }

    CUDA_CALLABLE vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    CUDA_CALLABLE vec3 &operator-=(const vec3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    CUDA_CALLABLE vec3 &operator*=(float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    CUDA_CALLABLE vec3 &operator/=(float t)
    {
        return *this *= 1 / t;
    }

    CUDA_CALLABLE float length() const
    {
        return sqrt(length_squared());
    }

    CUDA_CALLABLE float length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions

CUDA_CALLABLE inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

CUDA_CALLABLE inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

CUDA_CALLABLE inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

CUDA_CALLABLE inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

CUDA_CALLABLE inline vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}

CUDA_CALLABLE inline vec3 operator/(const vec3 &v, float t)
{
    return (1 / t) * v;
}

CUDA_CALLABLE inline float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

CUDA_CALLABLE inline vec3 project(const vec3 &u, const vec3 &v)
{
    return (dot(u, v) / dot(v, v)) * v;
}

CUDA_CALLABLE inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

CUDA_CALLABLE inline vec3 unit_vector(const vec3 &v)
{
    return v / v.length();
}

#endif