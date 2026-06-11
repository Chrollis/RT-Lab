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
        double vfov,
        double aspect,
        double near_plane = 0.1,
        double far_plane = 1000.0);
    ray get_ray(double u, double v) const;

    homogeneous_transform view_matrix() const;
    homogeneous_transform projection_matrix() const;
    homogeneous_transform view_projection_matrix() const;

    void move_forward(double distance);
    void move_right(double distance);
    void move_up(double distance);
    void rotate_yaw(double angle_deg);
    void rotate_pitch(double angle_deg);
    void rotate_roll(double angle_deg);

    void set_pose(
        const euclidean_coordinate& new_origin,
        const euclidean_coordinate& new_lookat);
    void set_lookat(const euclidean_coordinate& target);
    void set_fov(double new_vfov);
    void set_aspect(double new_aspect);
    void set_near_far(double near, double far);

private:
    euclidean_coordinate origin_;
    euclidean_coordinate lookat_;
    euclidean_coordinate world_up_;
    double vfov_;
    double aspect_;
    double near_plane_, far_plane_;

    mutable bool cache_valid_;
    mutable euclidean_coordinate forward_, right_, up_;

    void update_axes() const;
    void update_lookat_from_axes();
};
}  // namespace chrray