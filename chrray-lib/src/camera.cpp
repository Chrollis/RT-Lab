#include <camera.h>
#include <cmath>

namespace chrray {

camera::camera(
    const euclidean_coordinate& origin,
    const euclidean_coordinate& lookat,
    const euclidean_coordinate& world_up,
    float vfov,
    float aspect,
    float near_plane,
    float far_plane)
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
ray camera::get_ray(float u, float v) const {
    if (!cache_valid_) update_axes();
    float half_height = near_plane_ * std::tan(vfov_ * 0.5f * pi / 180.0f);
    float viewport_height = 2.0f * half_height;
    float viewport_width = aspect_ * viewport_height;
    float offset_x = (u - 0.5f) * viewport_width;
    float offset_y = (0.5f - v) * viewport_height;
    euclidean_coordinate direction =
        forward_ * near_plane_ + right_ * offset_x + up_ * offset_y;
    return ray(origin_, direction.normalize());
}

void camera::move_forward(float distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + forward_ * distance;
    lookat_ = origin_ + forward_;
}
void camera::move_right(float distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + right_ * distance;
    lookat_ = origin_ + forward_;
}
void camera::move_up(float distance) {
    if (!cache_valid_) update_axes();
    origin_ = origin_ + up_ * distance;
    lookat_ = origin_ + forward_;
}
void camera::rotate_yaw(float angle_deg) {
    if (!cache_valid_) update_axes();
    float rad = angle_deg * pi / 180.0f;
    quaternion q = quaternion::from_axis_angle(world_up_, rad);
    forward_ = q.rotate_vector(forward_);
    right_ = forward_.cross(up_).normalize();
    up_ = right_.cross(forward_).normalize();
    lookat_ = origin_ + forward_;
    cache_valid_ = true;
}
void camera::rotate_pitch(float angle_deg) {
    if (!cache_valid_) update_axes();
    float rad = angle_deg * pi / 180.0f;
    quaternion q = quaternion::from_axis_angle(right_, rad);
    forward_ = q.rotate_vector(forward_);
    up_ = right_.cross(forward_).normalize();
    lookat_ = origin_ + forward_;
    cache_valid_ = true;
}
void camera::rotate_roll(float angle_deg) {
    if (!cache_valid_) update_axes();
    float rad = angle_deg * pi / 180.0f;
    quaternion q = quaternion::from_axis_angle(forward_, rad);
    up_ = q.rotate_vector(up_);
    right_ = forward_.cross(up_).normalize();
    up_ = right_.cross(forward_).normalize();
    lookat_ = origin_ + forward_;
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
void camera::set_fov(float new_vfov) {
    vfov_ = new_vfov;
}
void camera::set_aspect(float new_aspect) {
    aspect_ = new_aspect;
}
void camera::set_near_far(float near, float far) {
    near_plane_ = near;
    far_plane_ = far;
}
const euclidean_coordinate& camera::origin() const {
    return origin_;
}
const euclidean_coordinate& camera::lookat() const {
    return lookat_;
}
void camera::reset_up() {
    if (!cache_valid_) update_axes();
    right_ = forward_.cross(world_up_).normalize();
    up_ = right_.cross(forward_).normalize();
    cache_valid_ = true;
}
homogeneous_transform camera::view_matrix() const {
    if (!cache_valid_) update_axes();
    homogeneous_transform V;
    V(0, 0) = right_.x();
    V(0, 1) = right_.y();
    V(0, 2) = right_.z();
    V(0, 3) = -right_.dot(origin_);
    V(1, 0) = up_.x();
    V(1, 1) = up_.y();
    V(1, 2) = up_.z();
    V(1, 3) = -up_.dot(origin_);
    V(2, 0) = -forward_.x();
    V(2, 1) = -forward_.y();
    V(2, 2) = -forward_.z();
    V(2, 3) = forward_.dot(origin_);
    V(3, 0) = 0;
    V(3, 1) = 0;
    V(3, 2) = 0;
    V(3, 3) = 1;
    return V;
}

homogeneous_transform camera::projection_matrix() const {
    float f = 1.0f / tanf(vfov_ * 0.5f * pi / 180.0f);
    float aspect = aspect_;
    float n = near_plane_;
    float f_plane = far_plane_;
    homogeneous_transform P;
    P(0, 0) = f / aspect;
    P(1, 1) = f;
    P(2, 2) = -(f_plane + n) / (f_plane - n);
    P(2, 3) = -(2.0f * f_plane * n) / (f_plane - n);
    P(3, 2) = -1.0f;
    return P;
}
}  // namespace chrray