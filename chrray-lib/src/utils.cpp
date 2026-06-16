#include <utils.h>
#include <chrono>
#include <thread>

namespace chrray {
static thread_local uint32_t x = 123456789;
static thread_local uint32_t y = 362436069;
static thread_local uint32_t z = 521288629;
static thread_local uint32_t w = 88675123;

void init_random() {
    uint64_t t = std::chrono::steady_clock::now().time_since_epoch().count();
    t ^= std::hash<std::thread::id>{}(std::this_thread::get_id());
    x = static_cast<uint32_t>(t);
    y = 362436069;
    z = 521288629;
    w = 88675123;
}
uint32_t xorshift() {
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = w ^ (w >> 19) ^ t ^ (t >> 8);
    return w;
}
float random_float() {
    return xorshift() / 4294967296.0f;
}
float random_float(float min, float max) {
    return min + (max - min) * random_float();
}

euclidean_coordinate random_in_unit_sphere() {
    while (true) {
        auto p = euclidean_coordinate(
            random_float(-1, 1), random_float(-1, 1), random_float(-1, 1));
        if (p.x() * p.x() + p.y() * p.y() + p.z() * p.z() < 1.0f) return p;
    }
}
euclidean_coordinate random_on_unit_sphere() {
    return random_in_unit_sphere().normalize();
}
euclidean_coordinate random_on_hemisphere(const euclidean_coordinate& normal) {
    auto on_sphere = random_on_unit_sphere();
    if (on_sphere.dot(normal) > 0.0f)
        return on_sphere;
    else
        return -on_sphere;
}
euclidean_coordinate random_cosine_direction() {
    float r1 = random_float(), r2 = random_float();
    float phi = 2.0f * pi * r1;
    float cos_theta = sqrtf(r2),
          sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    return euclidean_coordinate(
        sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

euclidean_coordinate reflect(
    const euclidean_coordinate& v, const euclidean_coordinate& n) {
    return v - 2.0f * v.dot(n) * n;
}
bool refract(
    const euclidean_coordinate& v,
    const euclidean_coordinate& n,
    float ni_over_nt,
    euclidean_coordinate& refracted) {
    float dt = v.dot(n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt * (v - n * dt) - sqrtf(discriminant) * n;
        return true;
    }
    return false;
}
float schlick(float cosine, float refraction_index) {
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

aabb::aabb() : min(0, 0, 0), max(0, 0, 0) {}
aabb::aabb(const euclidean_coordinate& a, const euclidean_coordinate& b)
    : min(a), max(b) {}
bool aabb::hit_interval(
    const euclidean_coordinate& origin,
    const euclidean_coordinate& dir,
    float t_min,
    float t_max,
    float& t_entry,
    float& t_exit) const {
    float t0 = t_min, t1 = t_max;
    {
        float inv_d = 1.0f / dir.x();
        if (dir.x() == 0.0f) {
            if (origin.x() < min.x() || origin.x() > max.x()) return false;
        } else {
            float t_near = inv_d * (min.x() - origin.x());
            float t_far = inv_d * (max.x() - origin.x());
            if (inv_d < 0.0f) std::swap(t_near, t_far);
            t0 = std::max(t_near, t0);
            t1 = std::min(t_far, t1);
            if (t0 > t1) return false;
        }
    }
    {
        float inv_d = 1.0f / dir.y();
        if (dir.y() == 0.0f) {
            if (origin.y() < min.y() || origin.y() > max.y()) return false;
        } else {
            float t_near = inv_d * (min.y() - origin.y());
            float t_far = inv_d * (max.y() - origin.y());
            if (inv_d < 0.0f) std::swap(t_near, t_far);
            t0 = std::max(t_near, t0);
            t1 = std::min(t_far, t1);
            if (t0 > t1) return false;
        }
    }
    {
        float inv_d = 1.0f / dir.z();
        if (dir.z() == 0.0f) {
            if (origin.z() < min.z() || origin.z() > max.z()) return false;
        } else {
            float t_near = inv_d * (min.z() - origin.z());
            float t_far = inv_d * (max.z() - origin.z());
            if (inv_d < 0.0f) std::swap(t_near, t_far);
            t0 = std::max(t_near, t0);
            t1 = std::min(t_far, t1);
            if (t0 > t1) return false;
        }
    }
    t_entry = t0;
    t_exit = t1;
    return true;
}
bool aabb::hit(
    const euclidean_coordinate& origin,
    const euclidean_coordinate& dir,
    float t_min,
    float t_max) const {
    float temp0 = 0.0f, temp1 = 0.0f;
    return hit_interval(origin, dir, t_min, t_max, temp0, temp1);
}
float aabb::surface_area() const {
    auto dp = max - min;
    return 2.0f * (dp.x() * dp.y() + dp.y() * dp.z() + dp.z() * dp.x());
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
    float u,
    float v) {
    float w = 1.0f - u - v;
    return v0 * w + v1 * u + v2 * v;
}
}  // namespace chrray