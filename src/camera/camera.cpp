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
    
void Camera::handle_event(const SDL_Event event, Planet* bodies)
{
    if (event.type == SDL_MOUSEWHEEL)
    {
        lock_radius += event.wheel.y * 10;
        printf("camera_center.z: %f\n", center.z());
    }

    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.sym == SDLK_l)
        {
            lock_state++;
            if (lock_state > NUM_BODIES) lock_state = 0;
                printf("Camera lock: %s\n", lock_state ? "ON" : "OFF");
            return;
        }
        vec3 lock_pos = vec3(0, 0, 0);
        if (lock_state == 0) {
            lock_pos = get_center_of_mass(bodies);
        }
        else {
            lock_pos = bodies[lock_state - 1].pos;
        }

        switch (event.key.keysym.sym)
        {
        case SDLK_w:
            lock_phi += 0.1;
            break;
        case SDLK_s:
            lock_phi -= 0.1;
            break;
        case SDLK_a:
            lock_theta += 0.1;
            break;
        case SDLK_d:
            lock_theta -= 0.1;
            break;
        default:
            break;
        }
        center = lock_pos + vec3(
            lock_radius * sin(lock_phi) * cos(lock_theta),
            lock_radius * sin(lock_phi) * sin(lock_theta),
            lock_radius * cos(lock_phi)
        );
    }
}

void Camera::move(const vec3 &offset)
{
    center += offset;
    update_viewport();
}

void Camera::zoom(double delta) {
    if (lock){
        lock_radius += delta;
        lock_radius = std::max(1.0, lock_radius);
        return;
    } else {
        move(vec3(0, 0, delta));
    }
    update_viewport();
}

void Camera::lock_on(vec3 pos)
{
    center = pos + vec3(0, 0, -lock_radius);
    update_viewport();
}