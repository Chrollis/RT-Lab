#include <camera.h>
#include <cmath>

namespace chrray {

camera::camera(
    const euclidean_coordinate& origin,
    const euclidean_coordinate& lookat,
    const euclidean_coordinate& world_up,
    double vfov,
    double aspect,
    double near_plane,
    double far_plane)
    : origin_(origin),
      lookat_(lookat),
      world_up_(world_up),
      vfov_(vfov),
      aspect_(aspect),
      near_plane_(near_plane),
      far_plane_(far_plane),
      cache_valid_(false) {}

void camera::update_axes() const {
    forward_ = (lookat_ - origin_).normalize();
    right_ = forward_.cross(world_up_).normalize();
    up_ = right_.cross(forward_).normalize();
    cache_valid_ = true;
}
void camera::update_lookat_from_axes() {
    lookat_ = origin_ + forward_;
}
ray camera::get_ray(double u, double v) const {
    if (!cache_valid_) update_axes();
    double half_height = std::tan(vfov_ * 0.5 * pi / 180.0);
    double viewport_height = 2.0 * half_height;
    double viewport_width = aspect_ * viewport_height;
    double offset_x = (u - 0.5) * viewport_width;
    double offset_y = (0.5 - v) * viewport_height;
    euclidean_coordinate direction =
        forward_ + right_ * offset_x + up_ * offset_y;
    return ray(origin_, direction.normalize());
}

homogeneous_transform camera::view_matrix() const {
    if (!cache_valid_) update_axes();
    homogeneous_transform view;
    view(0, 0) = right_.x();
    view(0, 1) = right_.y();
    view(0, 2) = right_.z();
    view(0, 3) = -right_.dot(origin_);
    view(1, 0) = up_.x();
    view(1, 1) = up_.y();
    view(1, 2) = up_.z();
    view(1, 3) = -up_.dot(origin_);
    view(2, 0) = forward_.x();
    view(2, 1) = forward_.y();
    view(2, 2) = forward_.z();
    view(2, 3) = -forward_.dot(origin_);
    view(3, 0) = 0.0;
    view(3, 1) = 0.0;
    view(3, 2) = 0.0;
    view(3, 3) = 1.0;
    return view;
}
homogeneous_transform camera::projection_matrix() const {
    double f = 1.0 / std::tan(vfov_ * 0.5 * pi / 180.0);
    double n = near_plane_, f_ = far_plane_;
    double range_inv = 1.0 / (n - f_);
    homogeneous_transform proj;
    proj(0, 0) = f / aspect_;
    proj(1, 1) = f;
    proj(2, 2) = (n + f_) * range_inv;
    proj(2, 3) = 2.0 * n * f_ * range_inv;
    proj(3, 2) = -1.0;
    proj(3, 3) = 0.0;
    return proj;
}
homogeneous_transform camera::view_projection_matrix() const {
    return projection_matrix() * view_matrix();
}
void camera::move_forward(double distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + forward_ * distance;
    update_lookat_from_axes();
    cache_valid_ = false;
}
void camera::move_right(double distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + right_ * distance;
    update_lookat_from_axes();
    cache_valid_ = false;
}
void camera::move_up(double distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + up_ * distance;
    update_lookat_from_axes();
    cache_valid_ = false;
}
void camera::rotate_yaw(double angle_deg) {
    double rad = angle_deg * pi / 180.0;
    quaternion q =
        quaternion::from_axis_angle(euclidean_coordinate(0, 1, 0), rad);
    forward_ = q.rotate_vector(forward_);
    right_ = forward_.cross(world_up_).normalize();
    up_ = right_.cross(forward_).normalize();
    update_lookat_from_axes();
    cache_valid_ = true;
}
void camera::rotate_pitch(double angle_deg) {
    if (!cache_valid_) update_axes();
    double rad = angle_deg * pi / 180.0;
    quaternion q = quaternion::from_axis_angle(right_, rad);
    forward_ = q.rotate_vector(forward_);
    up_ = right_.cross(forward_).normalize();
    update_lookat_from_axes();
    cache_valid_ = true;
}
void camera::rotate_roll(double angle_deg) {
    if (!cache_valid_) update_axes();
    double rad = angle_deg * pi / 180.0;
    quaternion q = quaternion::from_axis_angle(forward_, rad);
    up_ = q.rotate_vector(up_);
    right_ = forward_.cross(up_).normalize();
    up_ = right_.cross(forward_).normalize();
    update_lookat_from_axes();
    cache_valid_ = true;
}
void camera::set_pose(
    const euclidean_coordinate& new_origin,
    const euclidean_coordinate& new_lookat) {
    origin_ = new_origin;
    lookat_ = new_lookat;
    cache_valid_ = false;
}
void camera::set_lookat(const euclidean_coordinate& target) {
    lookat_ = target;
    cache_valid_ = false;
}
void camera::set_fov(double new_vfov) {
    vfov_ = new_vfov;
}
void camera::set_aspect(double new_aspect) {
    aspect_ = new_aspect;
}
void camera::set_near_far(double near, double far) {
    near_plane_ = near;
    far_plane_ = far;
}
}  // namespace chrray