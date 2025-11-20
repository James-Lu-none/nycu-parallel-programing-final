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
    
void Camera::handle_event(const SDL_Event event)
{
    if (event.type == SDL_MOUSEWHEEL)
    {
        if (event.wheel.y > 0)
        {
            zoom(10);
        }
        else if (event.wheel.y < 0)
        {
            zoom(-10);
        }
        printf("camera_center.z: %f\n", center.z());
    }

    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.sym == SDLK_l)
        {
            lock_to_com = !lock_to_com;
            printf("Camera lock_to_com: %s\n", lock_to_com ? "ON" : "OFF");
            return;
        }

        if (lock_to_com)
        {
            return;
        }
        else
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_w:
                move(vec3(0, 10, 0));
                break;
            case SDLK_s:
                move(vec3(0, -10, 0));
                break;
            case SDLK_a:
                move(vec3(-10, 0, 0));
                break;
            case SDLK_d:
                move(vec3(10, 0, 0));
                break;
            default:
                break;
            }
        }
    }
}

void Camera::move(const vec3 &offset)
{
    center += offset;
    update_viewport();
}

void Camera::zoom(double delta) {
    if (lock_to_com){
        lock_radius += delta;
        lock_radius = std::max(1.0, lock_radius);
        return;
    } else {
        focal_length += delta * 0.1;
    }
    update_viewport();
}

void Camera::lock_com(vec3 com_pos)
{
    center = com_pos + vec3(0, 0, -lock_radius);
    update_viewport();
}