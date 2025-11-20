#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "vec3.hpp"

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

    void update_viewport();

    Camera(double focal_len = 1.0, vec3 center = vec3(0, 0, -10), double viewport_height = 2.0);

    void move(const vec3 &offset);
    void move_forward(double distance);
    void move_backward(double distance);
    void move_left(double distance);
    void move_right(double distance);
    void move_up(double distance);
    void move_down(double distance);

    void zoom(double delta);
};

#endif