#pragma once
#include <matrix.h>
#include <object.h>

namespace chrray {
class camera {
public:
    camera(
        const euclidean_coordinate& origin,
        const euclidean_coordinate& lookat,
        const euclidean_coordinate& world_up,
        float vfov,
        float aspect,
        float near_plane = 0.1f,
        float far_plane = 1000.0f);
    ray get_ray(float u, float v) const;

    void move_forward(float distance);
    void move_right(float distance);
    void move_up(float distance);
    void rotate_yaw(float angle_deg);
    void rotate_pitch(float angle_deg);
    void rotate_roll(float angle_deg);
    void reset_up();

    void set_pose(
        const euclidean_coordinate& new_origin,
        const euclidean_coordinate& new_lookat);
    void set_lookat(const euclidean_coordinate& target);
    void set_fov(float new_vfov);
    void set_aspect(float new_aspect);
    void set_near_far(float near, float far);

    const euclidean_coordinate& origin() const;
    const euclidean_coordinate& lookat() const;

    homogeneous_transform view_matrix() const;
    homogeneous_transform projection_matrix() const;

private:
    euclidean_coordinate origin_;
    euclidean_coordinate lookat_;
    euclidean_coordinate world_up_;
    float vfov_;
    float aspect_;
    float near_plane_, far_plane_;

    mutable bool cache_valid_;
    mutable euclidean_coordinate forward_, right_, up_;

    void update_axes() const;
};
}  // namespace chrray