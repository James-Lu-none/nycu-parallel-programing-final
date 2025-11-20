#include "camera.hpp"
#include "config.hpp"

void Camera::update_viewport()
{
    viewport_u = vec3(viewport_width, 0, 0);
    viewport_v = vec3(0, viewport_height, 0);
    pixel_delta_u = viewport_u / WIDTH;
    pixel_delta_v = viewport_v / HEIGHT;
    viewport_bottom_left = center + vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_bottom_left + 0.5 * (pixel_delta_u + pixel_delta_v);
}

Camera::Camera(double focal_len, vec3 center, double viewport_height)
    : center(center), focal_length(focal_len), viewport_height(viewport_height)
{
    viewport_width = viewport_height * (double(WIDTH) / HEIGHT);
    update_viewport();
}

void Camera::move(const vec3 &offset)
{
    center = center + offset;
    update_viewport();
}

void Camera::move_up(double distance) { move(vec3(0, distance, 0)); }
void Camera::move_down(double distance) { move(vec3(0, -distance, 0)); }
void Camera::move_left(double distance) { move(vec3(-distance, 0, 0)); }
void Camera::move_right(double distance) { move(vec3(distance, 0, 0)); }
void Camera::zoom(double delta) { move(vec3(0, 0, delta)); }