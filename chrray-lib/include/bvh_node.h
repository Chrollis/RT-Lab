#pragma once
#include <object.h>
#include <vector>

namespace chrray {

class bvh_node : public hittable {
public:
    bvh_node(
        std::vector<std::shared_ptr<hittable>>& objects,
        size_t start,
        size_t end,
        size_t leaf_size = 4);

    virtual bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual aabb bounding_box() const override;
    virtual bool any_hit(const ray& r, float t_min, float t_max) const override;

private:
    std::shared_ptr<bvh_node> left_;
    std::shared_ptr<bvh_node> right_;
    aabb box_;

    std::vector<std::shared_ptr<hittable>> leaf_objects_;

    bool intersect_node(
        const ray& r, float t_min, float t_max, hit_record& rec) const;
    bool any_hit_node(const ray& r, float t_min, float t_max) const;
};

}  // namespace chrray