#include <object.h>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace chrray {
ray::ray() : origin_(0, 0, 0), direction_(0, 0, 1) {}
ray::ray(
    const euclidean_coordinate& origin, const euclidean_coordinate& direction)
    : origin_(origin), direction_(direction.normalize()) {}
const euclidean_coordinate& ray::origin() const {
    return origin_;
}
const euclidean_coordinate& ray::direction() const {
    return direction_;
}
euclidean_coordinate ray::at(float t) const {
    return origin_ + t * direction_;
}

hit_record::hit_record() : t(0.0f), u(0.0f), v(0.0f) {}

sphere::sphere(
    const euclidean_coordinate& center,
    float radius,
    std::shared_ptr<material> mat)
    : center_(center), radius_(radius), mat_(mat) {}

bool sphere::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    euclidean_coordinate oc = r.origin() - center_;
    float a = r.direction().dot(r.direction());
    float b = 2.0f * oc.dot(r.direction());
    float c = oc.dot(oc) - radius_ * radius_;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < -eps) return false;
    if (discriminant < 0.0f) discriminant = 0.0f;

    float sqrt_d = std::sqrtf(discriminant);
    float t = (-b - sqrt_d) / (2.0f * a);
    if (t < t_min || t > t_max) {
        t = (-b + sqrt_d) / (2.0f * a);
        if (t < t_min || t > t_max) return false;
    }
    rec.t = t;
    rec.p = r.at(t);
    rec.n = (rec.p - center_) / radius_;
    if (rec.n.dot(r.direction()) > eps) {
        rec.n = rec.n * (-1.0f);
    }
    rec.mat = mat_;
    float theta = acos(-rec.n.y());
    float phi = atan2(-rec.n.z(), rec.n.x()) + pi;
    rec.u = phi / (2 * pi);
    rec.v = theta / pi;
    return true;
}
aabb sphere::bounding_box() const {
    euclidean_coordinate delta(radius_, radius_, radius_);
    return aabb(center_ - delta, center_ + delta);
}
bool sphere::any_hit(const ray& r, float t_min, float t_max) const {
    euclidean_coordinate oc = r.origin() - center_;
    float a = r.direction().dot(r.direction());
    float b = 2.0f * oc.dot(r.direction());
    float c = oc.dot(oc) - radius_ * radius_;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < -eps) return false;
    if (discriminant < 0.0f) discriminant = 0.0f;

    float sqrt_d = std::sqrtf(discriminant);
    float t = (-b - sqrt_d) / (2.0f * a);
    if (t < t_min || t > t_max) {
        t = (-b + sqrt_d) / (2.0f * a);
        if (t < t_min || t > t_max) return false;
    }
    return true;
}

triangle::triangle(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    std::shared_ptr<material> mat)
    : v0_(v0), v1_(v1), v2_(v2), mat_(mat) {
    euclidean_coordinate e1 = v1_ - v0_;
    euclidean_coordinate e2 = v2_ - v0_;
    normal_ = e1.cross(e2).normalize();
}

bool triangle::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    euclidean_coordinate e1 = v1_ - v0_;
    euclidean_coordinate e2 = v2_ - v0_;
    euclidean_coordinate pvec = r.direction().cross(e2);
    float det = e1.dot(pvec);
    if (fabs(det) < eps) return false;

    float inv_det = 1.0f / det;
    euclidean_coordinate tvec = r.origin() - v0_;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    euclidean_coordinate qvec = tvec.cross(e1);
    float v = r.direction().dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = e2.dot(qvec) * inv_det;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.n = normal_;
    if (rec.n.dot(r.direction()) > 0) {
        rec.n = rec.n * (-1.0f);
    }
    rec.mat = mat_;
    rec.u = u;
    rec.v = v;
    return true;
}
aabb triangle::bounding_box() const {
    euclidean_coordinate min(
        std::min({v0_.x(), v1_.x(), v2_.x()}),
        std::min({v0_.y(), v1_.y(), v2_.y()}),
        std::min({v0_.z(), v1_.z(), v2_.z()}));
    euclidean_coordinate max(
        std::max({v0_.x(), v1_.x(), v2_.x()}),
        std::max({v0_.y(), v1_.y(), v2_.y()}),
        std::max({v0_.z(), v1_.z(), v2_.z()}));
    return aabb(min, max);
}
bool triangle::any_hit(const ray& r, float t_min, float t_max) const {
    euclidean_coordinate e1 = v1_ - v0_;
    euclidean_coordinate e2 = v2_ - v0_;
    euclidean_coordinate pvec = r.direction().cross(e2);
    float det = e1.dot(pvec);
    if (fabs(det) < eps) return false;

    float inv_det = 1.0f / det;
    euclidean_coordinate tvec = r.origin() - v0_;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    euclidean_coordinate qvec = tvec.cross(e1);
    float v = r.direction().dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = e2.dot(qvec) * inv_det;
    return (t >= t_min && t <= t_max);
}

plane::plane(
    const euclidean_coordinate& point,
    const euclidean_coordinate& normal,
    std::shared_ptr<material> mat)
    : point_(point), normal_(normal.normalize()), mat_(mat) {}

bool plane::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    float denom = r.direction().dot(normal_);
    if (fabs(denom) < eps) return false;

    float t = (point_ - r.origin()).dot(normal_) / denom;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.n = normal_;
    if (rec.n.dot(r.direction()) > 0) {
        rec.n = rec.n * (-1.0f);
    }
    rec.mat = mat_;
    rec.u = 0.0f;
    rec.v = 0.0f;
    return true;
}
aabb plane::bounding_box() const {
    const float half_size = 1e4;
    return aabb(
        euclidean_coordinate(-half_size, -half_size, -half_size),
        euclidean_coordinate(half_size, half_size, half_size));
}
bool plane::any_hit(const ray& r, float t_min, float t_max) const {
    float denom = r.direction().dot(normal_);
    if (fabs(denom) < eps) return false;
    float t = (point_ - r.origin()).dot(normal_) / denom;
    return (t >= t_min && t <= t_max);
}

mesh::mesh(const std::filesystem::path& filepath, std::shared_ptr<material> mat)
    : mat_(mat) {
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error(
            "Cannot open .obj file from " + filepath.string());

    std::vector<euclidean_coordinate> vertices;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        } else if (prefix == "f") {
            std::vector<int> idxs;
            std::string token;
            while (iss >> token) {
                size_t slash = token.find('/');
                int idx = std::stoi(token.substr(0, slash));
                idxs.push_back(idx);
            }
            for (size_t i = 2; i < idxs.size(); ++i) {
                int i0 = idxs[0] - 1;
                int i1 = idxs[i - 1] - 1;
                int i2 = idxs[i] - 1;
                if (i0 >= 0 && i1 >= 0 && i2 >= 0 && i0 < vertices.size() &&
                    i1 < vertices.size() && i2 < vertices.size()) {
                    triangles_.push_back(
                        std::make_unique<triangle>(
                            vertices[i0], vertices[i1], vertices[i2], mat_));
                }
            }
        }
    }
    if (!triangles_.empty()) {
        bbox_ = triangles_[0]->bounding_box();
        for (size_t i = 1; i < triangles_.size(); ++i) {
            bbox_ = surrounding_box(bbox_, triangles_[i]->bounding_box());
        }
    }
}
bool mesh::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    bool hit_anything = false;
    float closest_so_far = t_max;
    hit_record temp_rec;
    for (const auto& tri : triangles_) {
        if (tri->intersect(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}
aabb mesh::bounding_box() const {
    return bbox_;
}
bool mesh::any_hit(const ray& r, float t_min, float t_max) const {
    for (const auto& tri : triangles_) {
        if (tri->any_hit(r, t_min, t_max)) return true;
    }
    return false;
}
}  // namespace chrray