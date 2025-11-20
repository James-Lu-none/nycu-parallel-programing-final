#include "camera.hpp"
#include "config.hpp"

void Camera::update_view(Planet *bodies)
{
    if (lock_state == 0)
    {
        lock_pos = get_center_of_mass(bodies);
    }
    else
    {
        lock_pos = bodies[lock_state - 1].pos;
    }

    offset = vec3(
        lock_radius * sin(lock_phi) * cos(lock_theta),
        lock_radius * sin(lock_phi) * sin(lock_theta),
        lock_radius * cos(lock_phi)
    );

    printf("Camera lock position: (%.2f, %.2f, %.2f)\n", lock_pos.x(), lock_pos.y(), lock_pos.z());
    printf("Camera offset: (%.2f, %.2f, %.2f)\n", offset.x(), offset.y(), offset.z());

    center = lock_pos + offset;
    vec3 unit_dir = unit_vector(-offset);
    viewport_u = vec3(viewport_width, 0, 0);
    viewport_v = vec3(0, viewport_height, 0);
    vec3 focal_offset = vec3(0, 0, focal_length);

    pixel_delta_u = viewport_u / WIDTH;
    pixel_delta_v = viewport_v / HEIGHT;
    viewport_bottom_left = center + focal_offset - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_bottom_left + 0.5 * (pixel_delta_u + pixel_delta_v);
}

Camera::Camera(double focal_len, vec3 center, double viewport_height)
    : center(center), focal_length(focal_len), viewport_height(viewport_height)
{
    viewport_width = viewport_height * (double(WIDTH) / HEIGHT);
}
    
void Camera::handle_event(const SDL_Event event)
{
    if (event.type == SDL_MOUSEWHEEL)
    {
        lock_radius += event.wheel.y * -10;
    }

    if (event.type == SDL_KEYDOWN)
    {
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
        case SDLK_l:
            lock_state++;
            if (lock_state > NUM_BODIES)
                lock_state = 0;
            printf("Camera lock: %d\n", lock_state);
            break;
        default:
            break;
        }
    }
}