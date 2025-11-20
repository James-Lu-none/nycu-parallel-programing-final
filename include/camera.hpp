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
    double focal_length;
    double viewport_height;
    double viewport_width;

    vec3 viewport_u;
    vec3 viewport_v;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 viewport_bottom_left;
    point3 pixel00_loc;

    bool lock = false;
    double lock_radius = 10.0;

    void update_viewport();

    Camera(double focal_len = 1.0, vec3 center = vec3(0, 0, -10), double viewport_height = 2.0);

    void handle_event(const SDL_Event event, Planet *bodies);
    void move(const vec3 &offset);
    void zoom(double delta);

    // Lock the camera center to the center of mass position
    void lock_on(vec3 pos);
};

#endif