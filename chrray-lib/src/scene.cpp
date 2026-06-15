#include <scene.h>
#include <algorithm>

namespace chrray {

scene::scene(accel_t accel_type) : accel_type_(accel_type) {
    rebuild_accel();
}
scene::~scene() = default;

void scene::add_object(std::shared_ptr<hittable> obj) {
    objects_.push_back(obj);
}
void scene::add_plane(std::shared_ptr<plane> pl) {
    planes_.push_back(pl);
}
void scene::add_light(std::shared_ptr<light> lit) {
    lights_.push_back(lit);
}
void scene::set_accel_type(accel_t type) {
    accel_type_ = type;
    rebuild_accel();
}
void scene::rebuild_accel() {
    bvh_root_ = nullptr;
    grid_root_ = nullptr;
    if (accel_type_ == accel_t::bvh && !objects_.empty()) {
        bvh_root_ = std::make_shared<bvh_node>(objects_, 0, objects_.size());
    } else if (accel_type_ == accel_t::uniform_grid && !objects_.empty()) {
        aabb world_box = objects_[0]->bounding_box();
        for (size_t i = 1; i < objects_.size(); ++i)
            world_box = surrounding_box(world_box, objects_[i]->bounding_box());
        int n = (int)std::cbrtf(objects_.size() / 4.0f);
        n = n > 1 ? n : 1;
        grid_root_ =
            std::make_shared<uniform_grid>(objects_, world_box, n, n, n);
    }
}
void scene::build_triangles() {
    global_triangles_.clear();
    for (auto& obj : objects_) {
        auto tris = obj->triangulate();
        global_triangles_.insert(
            global_triangles_.end(), tris.begin(), tris.end());
    }
}

bool scene::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    float closest_plane_t = t_max;
    hit_record plane_rec;
    for (const auto& pl : planes_) {
        hit_record temp;
        if (pl->intersect(r, t_min, closest_plane_t, temp)) {
            closest_plane_t = temp.t;
            plane_rec = temp;
        }
    }
    bool hit_plane = (closest_plane_t < t_max);

    float t_max_actual = hit_plane ? closest_plane_t : t_max;

    bool hit_obj = false;
    hit_record obj_rec;
    if (accel_type_ == accel_t::linear) {
        hit_obj = intersect_linear(r, t_min, t_max_actual, obj_rec);
    } else if (accel_type_ == accel_t::bvh && bvh_root_) {
        hit_obj = intersect_bvh(r, t_min, t_max_actual, obj_rec);
    } else if (accel_type_ == accel_t::uniform_grid && grid_root_) {
        hit_obj = intersect_grid(r, t_min, t_max_actual, obj_rec);
    } else {
        hit_obj = intersect_linear(r, t_min, t_max_actual, obj_rec);
    }

    if (hit_plane && hit_obj) {
        rec = (plane_rec.t < obj_rec.t) ? plane_rec : obj_rec;
        return true;
    } else if (hit_plane) {
        rec = plane_rec;
        return true;
    } else if (hit_obj) {
        rec = obj_rec;
        return true;
    }
    return false;
}

bool scene::any_hit(const ray& r, float t_min, float t_max) const {
    for (const auto& pl : planes_) {
        if (pl->any_hit(r, t_min, t_max)) return true;
    }
    if (accel_type_ == accel_t::linear)
        return any_hit_linear(r, t_min, t_max);
    else if (accel_type_ == accel_t::bvh && bvh_root_)
        return any_hit_bvh(r, t_min, t_max);
    else if (accel_type_ == accel_t::uniform_grid && grid_root_)
        return any_hit_grid(r, t_min, t_max);
    else
        return any_hit_linear(r, t_min, t_max);
}
const std::vector<std::shared_ptr<light>>& scene::get_lights() const {
    return lights_;
}
const std::vector<std::shared_ptr<hittable>>& scene::get_objects() const {
    return objects_;
}
const std::vector<std::shared_ptr<plane>>& scene::get_planes() const {
    return planes_;
}
const std::vector<Triangle>& scene::get_global_triangles() const {
    return global_triangles_;
}
bool scene::intersect_linear(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    bool hit = false;
    float closest = t_max;
    for (const auto& obj : objects_) {
        hit_record temp;
        if (obj->intersect(r, t_min, closest, temp)) {
            hit = true;
            closest = temp.t;
            rec = temp;
        }
    }
    return hit;
}

bool scene::intersect_bvh(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!bvh_root_) return false;
    return bvh_root_->intersect(r, t_min, t_max, rec);
}
bool scene::intersect_grid(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    return grid_root_ ? grid_root_->intersect(r, t_min, t_max, rec) : false;
}

bool scene::any_hit_linear(const ray& r, float t_min, float t_max) const {
    for (const auto& obj : objects_) {
        if (obj->any_hit(r, t_min, t_max)) return true;
    }
    return false;
}

bool scene::any_hit_bvh(const ray& r, float t_min, float t_max) const {
    if (!bvh_root_) return false;
    return bvh_root_->any_hit(r, t_min, t_max);
}
bool scene::any_hit_grid(const ray& r, float t_min, float t_max) const {
    return grid_root_ ? grid_root_->any_hit(r, t_min, t_max) : false;
}
}  // namespace chrray