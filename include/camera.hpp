#ifndef CAMERA_HPP
#define CAMERA_HPP
#include <SDL2/SDL.h>
#include "vec3.hpp"
#include "planet.hpp"
#include "config.hpp"

class Camera
{
public:
    point3 center;
    float focal_length;
    float viewport_height;
    float viewport_width;

    vec3 viewport_u;
    vec3 viewport_v;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 viewport_bottom_left;
    point3 pixel00_loc;

    vec3 lock_pos;
    vec3 offset;
    int lock_state = 0;
    float lock_phi = 0.0;
    float lock_theta = 0.0;
    float lock_radius = 10.0;

    void update_view(vector<Planet>& bodies);

    Camera(float focal_len = 1.0, vec3 center = vec3(0, 0, -10), float viewport_height = 2.0);

    void handle_event(const SDL_Event event);
};

#endif