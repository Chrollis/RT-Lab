#pragma once
#include <immintrin.h>
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

struct Triangle {
    euclidean_coordinate v0, v1, v2;
    euclidean_coordinate normal;
    std::shared_ptr<material> mat;
    Triangle(
        const euclidean_coordinate& v0,
        const euclidean_coordinate& v1,
        const euclidean_coordinate& v2,
        std::shared_ptr<material> mat);
};

struct ray8 {
    alignas(32) __m256 ox, oy, oz;
    alignas(32) __m256 dx, dy, dz;
    alignas(32) __m256 tmin, tmax;
    __m256i active;

    ray8() = default;
    ray8(const ray* rays, const __m256i& mask);
};

struct hit_record8 {
    alignas(32) __m256 t;
    alignas(32) __m256 px, py, pz;
    alignas(32) __m256 nx, ny, nz;
    alignas(32) __m256 u, v;
    alignas(32) int mat_id[8];
    __m256i active;

    hit_record8() = default;
    void clear() { active = _mm256_setzero_si256(); }
};

class hittable {
public:
    virtual ~hittable() = default;
    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    virtual aabb bounding_box() const = 0;
    virtual bool any_hit(const ray& r, float t_min, float t_max) const = 0;
    virtual std::vector<Triangle> triangulate() const;
    virtual void intersect8(const ray8& rays, hit_record8& recs) const;
    virtual __m256i any_hit8(const ray8& rays) const;

protected:
    int mat_index_ = 0;
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
    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;
    virtual std::vector<Triangle> triangulate() const override;
    virtual void intersect8(const ray8& rays, hit_record8& recs) const override;
    virtual __m256i any_hit8(const ray8& rays) const override;

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
    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;
    virtual std::vector<Triangle> triangulate() const override;
    virtual void intersect8(const ray8& rays, hit_record8& recs) const override;
    virtual __m256i any_hit8(const ray8& rays) const override;

private:
    Triangle tri_;
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
    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;
    virtual std::vector<Triangle> triangulate() const override;
    virtual void intersect8(const ray8& rays, hit_record8& recs) const override;
    virtual __m256i any_hit8(const ray8& rays) const override;

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
    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;
    virtual std::vector<Triangle> triangulate() const override;
    virtual void intersect8(const ray8& rays, hit_record8& recs) const override;
    virtual __m256i any_hit8(const ray8& rays) const override;

private:
    std::vector<std::unique_ptr<triangle>> triangles_;
    aabb bbox_;
    std::shared_ptr<material> mat_;
};
}  // namespace chrray