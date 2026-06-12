#pragma once
#include <matrix.h>
#include <object.h>
#include <memory>
#include <vector>

namespace chrray {
class uniform_grid : public hittable {
public:
    uniform_grid(
        const std::vector<std::shared_ptr<hittable>>& objects,
        const aabb& world_box,
        int nx,
        int ny,
        int nz);

    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;

    virtual aabb bounding_box() const override;

private:
    int nx_, ny_, nz_;
    aabb box_;
    euclidean_coordinate step_, inv_step_;
    std::vector<std::vector<std::shared_ptr<hittable>>> cells_;
};
}  // namespace chrray