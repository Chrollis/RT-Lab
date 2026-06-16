#pragma once
#include <matrix.h>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace chrray {
constexpr float pi = 3.14159267f;
constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float eps = 5e-4f;

void init_random(unsigned seed);
float random_float();
float random_float(float min, float max);

euclidean_coordinate random_in_unit_sphere();
euclidean_coordinate random_on_unit_sphere();
euclidean_coordinate random_on_hemisphere(const euclidean_coordinate& normal);
euclidean_coordinate random_cosine_direction();

euclidean_coordinate reflect(
    const euclidean_coordinate& v, const euclidean_coordinate& n);
bool refract(
    const euclidean_coordinate& v,
    const euclidean_coordinate& n,
    float ni_over_nt,
    euclidean_coordinate& refracted);
float schlick(float cosine, float refraction_index);

struct aabb {
    euclidean_coordinate min;
    euclidean_coordinate max;
    aabb();
    aabb(const euclidean_coordinate& a, const euclidean_coordinate& b);
    bool hit_interval(
        const euclidean_coordinate& origin,
        const euclidean_coordinate& dir,
        float t_min,
        float t_max,
        float& t_entry,
        float& t_exit) const;
    bool hit(
        const euclidean_coordinate& origin,
        const euclidean_coordinate& dir,
        float t_min,
        float t_max) const;
    float surface_area() const;
};
aabb surrounding_box(const aabb& box0, const aabb& box1);
euclidean_coordinate barycentric_interpolate(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    float u,
    float v);

}  // namespace chrray