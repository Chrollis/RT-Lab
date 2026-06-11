#include <utils.h>
#include <random>

namespace chrray {
static std::mt19937 rng;
static std::uniform_real_distribution<double> dist(0.0, 1.0);

void init_random() {
    rng.seed(std::random_device{}());
}
double random_double() {
    return dist(rng);
}
double random_double(double min, double max) {
    return min + (max - min) * dist(rng);
}

euclidean_coordinate random_in_unit_sphere() {
    while (true) {
        auto p = euclidean_coordinate(
            random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
        if (p.x() * p.x() + p.y() * p.y() + p.z() * p.z() < 1.0) return p;
    }
}
euclidean_coordinate random_on_unit_sphere() {
    return random_in_unit_sphere().normalize();
}
euclidean_coordinate random_on_hemisphere(const euclidean_coordinate& normal) {
    auto on_sphere = random_on_unit_sphere();
    if (on_sphere.dot(normal) > 0.0)
        return on_sphere;
    else
        return -on_sphere;
}
euclidean_coordinate random_cosine_direction() {
    double r1 = random_double(), r2 = random_double();
    double phi = 2.0 * pi * r1;
    double cos_theta = sqrt(r2), sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return euclidean_coordinate(
        sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

euclidean_coordinate reflect(
    const euclidean_coordinate& v, const euclidean_coordinate& n) {
    return v - 2.0 * v.dot(n) * n;
}
bool refract(
    const euclidean_coordinate& v,
    const euclidean_coordinate& n,
    double ni_over_nt,
    euclidean_coordinate& refracted) {
    double dt = v.dot(n);
    double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if (discriminant > 0.0) {
        refracted = ni_over_nt * (v - n * dt) - sqrt(discriminant) * n;
        return true;
    }
    return false;
}
double schlick(double cosine, double refraction_index) {
    double r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

aabb::aabb() : min(0, 0, 0), max(0, 0, 0) {}
aabb::aabb(const euclidean_coordinate& a, const euclidean_coordinate& b)
    : min(a), max(b) {}
bool aabb::hit(
    const euclidean_coordinate& origin,
    const euclidean_coordinate& dir,
    double t_min,
    double t_max) const {
    double t0 = t_min, t1 = t_max;
    for (int i = 0; i < 3; i++) {
        double inv_d = 1.0 / dir[i];
        double t_near = inv_d * (min[i] - origin[i]);
        double t_far = inv_d * (max[i] - origin[i]);
        if (inv_d < 0.0) std::swap(t_near, t_far);
        t0 = std::max(t_near, t0);
        t1 = std::min(t_far, t1);
        if (t0 > t1) return false;
    }
    return true;
}
double aabb::surface_area() const {
    auto dp = max - min;
    return 2.0 * (dp.x() * dp.y() + dp.y() * dp.z() + dp.z() * dp.x());
}
aabb surrounding_box(const aabb& box0, const aabb& box1) {
    euclidean_coordinate min(
        std::min(box0.min.x(), box1.min.x()),
        std::min(box0.min.y(), box1.min.y()),
        std::min(box0.min.z(), box1.min.z()));
    euclidean_coordinate max(
        std::max(box0.max.x(), box1.max.x()),
        std::max(box0.max.y(), box1.max.y()),
        std::max(box0.max.z(), box1.max.z()));
    return aabb(min, max);
}
euclidean_coordinate barycentric_interpolate(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    double u,
    double v) {
    double w = 1.0 - u - v;
    return v0 * w + v1 * u + v2 * v;
}
}  // namespace chrray