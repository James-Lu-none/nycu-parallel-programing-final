#ifndef RAY_H
#define RAY_H

#include "vec3.hpp"

class ray
{
public:
    CUDA_HOSTDEV ray() {}

    CUDA_HOSTDEV ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    CUDA_HOSTDEV const point3 &origin() const { return orig; }
    CUDA_HOSTDEV const vec3 &direction() const { return dir; }

    CUDA_HOSTDEV point3 at(float t) const
    {
        return orig + t * dir;
    }

private:
    point3 orig;
    vec3 dir;
};

#endif