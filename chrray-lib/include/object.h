#pragma once
#include <matrix.h>
#include <utils.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace chrray {
class ray {
public:
    ray();
    ray(const euclidean_coordinate& origin,
        const euclidean_coordinate& direction);

    const euclidean_coordinate& origin() const;
    const euclidean_coordinate& direction() const;

    euclidean_coordinate at(float t) const;

private:
    euclidean_coordinate origin_;
    euclidean_coordinate direction_;
};

class material;

struct hit_record {
    float t;
    euclidean_coordinate p;
    euclidean_coordinate n;
    std::shared_ptr<material> mat;
    float u, v;
    hit_record();
};

class hittable {
public:
    virtual ~hittable() = default;
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    virtual aabb bounding_box() const = 0;
    virtual bool any_hit(const ray& r, float t_min, float t_max) const;
};

class sphere : public hittable {
public:
    sphere(
        const euclidean_coordinate& center,
        float radius,
        std::shared_ptr<material> mat);
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual aabb bounding_box() const override;

private:
    euclidean_coordinate center_;
    float radius_;
    std::shared_ptr<material> mat_;
};

class triangle : public hittable {
public:
    triangle(
        const euclidean_coordinate& v0,
        const euclidean_coordinate& v1,
        const euclidean_coordinate& v2,
        std::shared_ptr<material> mat);
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual aabb bounding_box() const override;

private:
    euclidean_coordinate v0_, v1_, v2_;
    euclidean_coordinate normal_;
    std::shared_ptr<material> mat_;
};

class plane : public hittable {
public:
    plane(
        const euclidean_coordinate& point,
        const euclidean_coordinate& normal,
        std::shared_ptr<material> mat);
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual aabb bounding_box() const override;

private:
    euclidean_coordinate point_;
    euclidean_coordinate normal_;
    std::shared_ptr<material> mat_;
};

class mesh : public hittable {
public:
    mesh(const std::filesystem::path& filepath, std::shared_ptr<material> mat);
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual aabb bounding_box() const override;

private:
    std::vector<std::unique_ptr<hittable>> triangles_;
    aabb bbox_;
    std::shared_ptr<material> mat_;
};
}  // namespace chrray