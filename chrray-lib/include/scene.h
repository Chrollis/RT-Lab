#pragma once
#include <bvh_node.h>
#include <light.h>
#include <object.h>
#include <uniform_grid.h>
#include <memory>
#include <vector>

namespace chrray {

enum class accel_t {
    linear,
    bvh,
    uniform_grid,
};

class scene {
public:
    scene(accel_t accel_type = accel_t::linear);
    ~scene();

    void add_object(std::shared_ptr<hittable> obj);
    void add_plane(std::shared_ptr<plane> pl);
    void add_light(std::shared_ptr<light> lit);
    void set_accel_type(accel_t type);
    void rebuild_accel();

    bool intersect(
        const ray& r, float t_min, float t_max, hit_record& rec) const;
    bool any_hit(const ray& r, float t_min, float t_max) const;
    const std::vector<std::shared_ptr<light>>& get_lights() const;

private:
    std::vector<std::shared_ptr<hittable>> objects_;
    std::vector<std::shared_ptr<plane>> planes_;
    std::vector<std::shared_ptr<light>> lights_;

    accel_t accel_type_;
    std::shared_ptr<bvh_node> bvh_root_;
    std::shared_ptr<uniform_grid> grid_root_;

    bool intersect_linear(
        const ray& r, float t_min, float t_max, hit_record& rec) const;
    bool intersect_bvh(
        const ray& r, float t_min, float t_max, hit_record& rec) const;
    bool intersect_grid(
        const ray& r, float t_min, float t_max, hit_record& rec) const;

    bool any_hit_linear(const ray& r, float t_min, float t_max) const;
    bool any_hit_bvh(const ray& r, float t_min, float t_max) const;
    bool any_hit_grid(const ray& r, float t_min, float t_max) const;
};

}  // namespace chrray