#pragma once
#include <matrix.h>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace chrray {
constexpr double pi = 3.14159265358979323846;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double eps = 1e-6;

void init_random();
double random_double();
double random_double(double min, double max);

euclidean_coordinate random_in_unit_sphere();
euclidean_coordinate random_on_unit_sphere();
euclidean_coordinate random_on_hemisphere(const euclidean_coordinate& normal);
euclidean_coordinate random_cosine_direction();

euclidean_coordinate reflect(
    const euclidean_coordinate& v, const euclidean_coordinate& n);
bool refract(
    const euclidean_coordinate& v,
    const euclidean_coordinate& n,
    double ni_over_nt,
    euclidean_coordinate& refracted);
double schlick(double cosine, double refraction_index);

struct aabb {
    euclidean_coordinate min;
    euclidean_coordinate max;
    aabb();
    aabb(const euclidean_coordinate& a, const euclidean_coordinate& b);
    bool hit(
        const euclidean_coordinate& origin,
        const euclidean_coordinate& dir,
        double t_min,
        double t_max) const;
    double surface_area() const;
};
aabb surrounding_box(const aabb& box0, const aabb& box1);
euclidean_coordinate barycentric_interpolate(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    double u,
    double v);

}  // namespace chrray