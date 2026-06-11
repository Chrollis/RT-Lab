#include <bvh_node.h>
#include <algorithm>

namespace chrray {

static inline float center_on_axis(
    const std::shared_ptr<hittable>& obj, int axis) {
    aabb b = obj->bounding_box();
    return (b.min[axis] + b.max[axis]) * 0.5f;
}

bvh_node::bvh_node(
    std::vector<std::shared_ptr<hittable>>& objects,
    size_t start,
    size_t end,
    size_t leaf_size) {
    box_ = objects[start]->bounding_box();
    for (size_t i = start + 1; i < end; ++i)
        box_ = surrounding_box(box_, objects[i]->bounding_box());

    if (end - start <= leaf_size) {
        leaf_objects_.assign(objects.begin() + start, objects.begin() + end);
        left_ = right_ = nullptr;
        return;
    }

    float dx = box_.max.x() - box_.min.x();
    float dy = box_.max.y() - box_.min.y();
    float dz = box_.max.z() - box_.min.z();
    int axis = 0;
    if (dy > dx && dy > dz)
        axis = 1;
    else if (dz > dx && dz > dy)
        axis = 2;

    auto comparator = [axis](
                          const std::shared_ptr<hittable>& a,
                          const std::shared_ptr<hittable>& b) {
        return center_on_axis(a, axis) < center_on_axis(b, axis);
    };
    std::sort(objects.begin() + start, objects.begin() + end, comparator);

    size_t mid = start + (end - start) / 2;
    left_ = std::make_shared<bvh_node>(objects, start, mid, leaf_size);
    right_ = std::make_shared<bvh_node>(objects, mid, end, leaf_size);
}

bool bvh_node::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    return intersect_node(r, t_min, t_max, rec);
}

bool bvh_node::intersect_node(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!left_ && !right_) {
        bool hit = false;
        float closest = t_max;
        for (const auto& obj : leaf_objects_) {
            if (obj->intersect(r, t_min, closest, rec)) {
                hit = true;
                closest = rec.t;
            }
        }
        return hit;
    }

    float t_left_entry = t_min, t_left_exit = t_max;
    float t_right_entry = t_min, t_right_exit = t_max;
    bool hit_left_box = left_ ? left_->box_.hit_interval(
                                    r.origin(), r.direction(), t_min, t_max,
                                    t_left_entry, t_left_exit)
                              : false;
    bool hit_right_box = right_ ? right_->box_.hit_interval(
                                      r.origin(), r.direction(), t_min, t_max,
                                      t_right_entry, t_right_exit)
                                : false;

    const bvh_node* first = nullptr;
    const bvh_node* second = nullptr;
    if (hit_left_box && hit_right_box) {
        if (t_left_entry < t_right_entry) {
            first = left_.get();
            second = right_.get();
        } else {
            first = right_.get();
            second = left_.get();
        }
    } else if (hit_left_box) {
        first = left_.get();
    } else if (hit_right_box) {
        first = right_.get();
    } else {
        return false;
    }

    bool hit = false;
    if (first) {
        if (first->intersect_node(r, t_min, t_max, rec)) {
            hit = true;
            t_max = rec.t;
        }
    }

    if (second) {
        float second_entry =
            (second == left_.get()) ? t_left_entry : t_right_entry;
        if (!hit || second_entry < t_max) {
            hit_record rec2;
            if (second->intersect_node(r, t_min, t_max, rec2)) {
                if (!hit || rec2.t < rec.t) {
                    rec = rec2;
                    hit = true;
                }
            }
        }
    }
    return hit;
}

bool bvh_node::any_hit(const ray& r, float t_min, float t_max) const {
    return any_hit_node(r, t_min, t_max);
}

bool bvh_node::any_hit_node(const ray& r, float t_min, float t_max) const {
    if (!left_ && !right_) {
        for (const auto& obj : leaf_objects_) {
            if (obj->any_hit(r, t_min, t_max)) return true;
        }
        return false;
    }

    float t_left_entry = t_min, t_left_exit = t_max;
    float t_right_entry = t_min, t_right_exit = t_max;
    bool hit_left_box = left_ ? left_->box_.hit_interval(
                                    r.origin(), r.direction(), t_min, t_max,
                                    t_left_entry, t_left_exit)
                              : false;
    bool hit_right_box = right_ ? right_->box_.hit_interval(
                                      r.origin(), r.direction(), t_min, t_max,
                                      t_right_entry, t_right_exit)
                                : false;

    const bvh_node* first = nullptr;
    const bvh_node* second = nullptr;
    if (hit_left_box && hit_right_box) {
        if (t_left_entry < t_right_entry) {
            first = left_.get();
            second = right_.get();
        } else {
            first = right_.get();
            second = left_.get();
        }
    } else if (hit_left_box) {
        first = left_.get();
    } else if (hit_right_box) {
        first = right_.get();
    } else {
        return false;
    }

    if (first && first->any_hit_node(r, t_min, t_max)) return true;
    if (second && second->any_hit_node(r, t_min, t_max)) return true;
    return false;
}

aabb bvh_node::bounding_box() const {
    return box_;
}

}  // namespace chrray