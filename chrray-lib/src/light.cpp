#include <light.h>

namespace chrray {
point_light::point_light(
    const euclidean_coordinate& position,
    const color& intensity,
    double constant_atten,
    double linear_atten,
    double quad_atten)
    : position_(position),
      intensity_(intensity),
      const_atten_(constant_atten),
      lin_atten_(linear_atten),
      quad_atten_(quad_atten) {}

euclidean_coordinate point_light::direction(
    const euclidean_coordinate& hit_point) const {
    return (position_ - hit_point).normalize();
}
double point_light::distance(const euclidean_coordinate& hit_point) const {
    return (position_ - hit_point).length();
}
color point_light::intensity(const euclidean_coordinate& hit_point) const {
    double d = distance(hit_point);
    double att = const_atten_ + lin_atten_ * d + quad_atten_ * d * d;
    if (att == 0.0) return intensity_;
    return intensity_ * (1.0 / att);
}
ray point_light::shadow_ray(const euclidean_coordinate& hit_point) const {
    euclidean_coordinate dir = direction(hit_point);
    euclidean_coordinate origin = hit_point + dir * eps;
    return ray(origin, dir);
}

directional_light::directional_light(
    const euclidean_coordinate& direction, const color& intensity)
    : direction_(direction.normalize()), intensity_(intensity) {}
euclidean_coordinate directional_light::direction(
    const euclidean_coordinate& hit_point) const {
    return direction_;
}
double directional_light::distance(
    const euclidean_coordinate& hit_point) const {
    return inf;
}
color directional_light::intensity(
    const euclidean_coordinate& hit_point) const {
    return intensity_;
}
ray directional_light::shadow_ray(const euclidean_coordinate& hit_point) const {
    euclidean_coordinate dir = direction_;
    euclidean_coordinate origin = hit_point + dir * eps;
    return ray(origin, dir);
}
}  // namespace chrray