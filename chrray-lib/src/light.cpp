#include <light.h>

namespace chrray {
point_light::point_light(
    const euclidean_coordinate& position,
    const color& intensity,
    float constant_atten,
    float linear_atten,
    float quad_atten)
    : position_(position),
      intensity_(intensity),
      const_atten_(constant_atten),
      lin_atten_(linear_atten),
      quad_atten_(quad_atten) {}

euclidean_coordinate point_light::direction(
    const euclidean_coordinate& hit_point) const {
    return (position_ - hit_point).normalize();
}
float point_light::distance(const euclidean_coordinate& hit_point) const {
    return (position_ - hit_point).length();
}
color point_light::intensity(const euclidean_coordinate& hit_point) const {
    float d = distance(hit_point);
    float att = const_atten_ + lin_atten_ * d + quad_atten_ * d * d;
    if (att == 0.0f) return intensity_;
    return intensity_ * (1.0f / att);
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
float directional_light::distance(const euclidean_coordinate& hit_point) const {
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