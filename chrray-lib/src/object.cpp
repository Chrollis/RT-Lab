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
Triangle::Triangle(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    std::shared_ptr<material> mat)
    : v0(v0), v1(v1), v2(v2), mat(mat) {
    euclidean_coordinate e1 = v1 - v0;
    euclidean_coordinate e2 = v2 - v0;
    normal = e1.cross(e2).normalize();
}
std::vector<Triangle> hittable::triangulate() const {
    return {};
}

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
std::vector<Triangle> sphere::triangulate() const {
    std::vector<Triangle> tris;
    const int stacks = 16;
    const int slices = 32;
    for (int i = 0; i < stacks; ++i) {
        float theta1 = pi * i / stacks;
        float theta2 = pi * (i + 1) / stacks;
        for (int j = 0; j < slices; ++j) {
            float phi1 = 2 * pi * j / slices;
            float phi2 = 2 * pi * (j + 1) / slices;
            auto vert = [&](float theta, float phi) -> euclidean_coordinate {
                float x = sinf(theta) * cosf(phi);
                float y = cosf(theta);
                float z = sinf(theta) * sinf(phi);
                return center_ + radius_ * euclidean_coordinate(x, y, z);
            };
            euclidean_coordinate v00 = vert(theta1, phi1);
            euclidean_coordinate v01 = vert(theta1, phi2);
            euclidean_coordinate v10 = vert(theta2, phi1);
            euclidean_coordinate v11 = vert(theta2, phi2);
            auto add_triangle = [&](const euclidean_coordinate& a,
                                    const euclidean_coordinate& b,
                                    const euclidean_coordinate& c) {
                euclidean_coordinate normal = (b - a).cross(c - a);
                if (normal.length() > 1e-6f) {
                    tris.emplace_back(a, b, c, mat_);
                }
            };

            add_triangle(v00, v01, v10);
            add_triangle(v01, v11, v10);
        }
    }
    return tris;
}

triangle::triangle(
    const euclidean_coordinate& v0,
    const euclidean_coordinate& v1,
    const euclidean_coordinate& v2,
    std::shared_ptr<material> mat)
    : tri_(v0, v1, v2, mat) {}

bool triangle::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    euclidean_coordinate e1 = tri_.v1 - tri_.v0;
    euclidean_coordinate e2 = tri_.v2 - tri_.v0;
    euclidean_coordinate pvec = r.direction().cross(e2);
    float det = e1.dot(pvec);
    if (fabs(det) < eps) return false;

    float inv_det = 1.0f / det;
    euclidean_coordinate tvec = r.origin() - tri_.v0;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    euclidean_coordinate qvec = tvec.cross(e1);
    float v = r.direction().dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = e2.dot(qvec) * inv_det;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.n = tri_.normal;
    if (rec.n.dot(r.direction()) > 0) {
        rec.n = rec.n * (-1.0f);
    }
    rec.mat = tri_.mat;
    rec.u = u;
    rec.v = v;
    return true;
}
aabb triangle::bounding_box() const {
    euclidean_coordinate min(
        std::min({tri_.v0.x(), tri_.v1.x(), tri_.v2.x()}),
        std::min({tri_.v0.y(), tri_.v1.y(), tri_.v2.y()}),
        std::min({tri_.v0.z(), tri_.v1.z(), tri_.v2.z()}));
    euclidean_coordinate max(
        std::max({tri_.v0.x(), tri_.v1.x(), tri_.v2.x()}),
        std::max({tri_.v0.y(), tri_.v1.y(), tri_.v2.y()}),
        std::max({tri_.v0.z(), tri_.v1.z(), tri_.v2.z()}));
    return aabb(min, max);
}
bool triangle::any_hit(const ray& r, float t_min, float t_max) const {
    euclidean_coordinate e1 = tri_.v1 - tri_.v0;
    euclidean_coordinate e2 = tri_.v2 - tri_.v0;
    euclidean_coordinate pvec = r.direction().cross(e2);
    float det = e1.dot(pvec);
    if (fabs(det) < eps) return false;

    float inv_det = 1.0f / det;
    euclidean_coordinate tvec = r.origin() - tri_.v0;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    euclidean_coordinate qvec = tvec.cross(e1);
    float v = r.direction().dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = e2.dot(qvec) * inv_det;
    return (t >= t_min && t <= t_max);
}
std::vector<Triangle> triangle::triangulate() const {
    std::vector<Triangle> tris = {tri_};
    return tris;
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
std::vector<Triangle> plane::triangulate() const {
    std::vector<Triangle> tris;
    const float half_size = 50.0f;
    const int grid = 40;

    euclidean_coordinate right = euclidean_coordinate(1, 0, 0);
    if (fabs(normal_.dot(right)) > 0.9999f)
        right = euclidean_coordinate(0, 0, 1);
    euclidean_coordinate forward = right.cross(normal_).normalize();
    right = normal_.cross(forward).normalize();

    euclidean_coordinate center = point_;
    euclidean_coordinate start =
        center - right * half_size - forward * half_size;
    float step = 2.0f * half_size / grid;

    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j) {
            euclidean_coordinate p00 =
                start + right * (i * step) + forward * (j * step);
            euclidean_coordinate p10 = p00 + right * step;
            euclidean_coordinate p01 = p00 + forward * step;
            euclidean_coordinate p11 = p00 + right * step + forward * step;
            tris.emplace_back(p00, p10, p11, mat_);
            tris.emplace_back(p00, p11, p01, mat_);
        }
    }
    return tris;
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
std::vector<Triangle> mesh::triangulate() const {
    std::vector<Triangle> tris;
    for (const auto& t : triangles_) {
        tris.push_back(t->triangulate()[0]);
    }
    return tris;
}
}  // namespace chrray