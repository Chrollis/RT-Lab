#include <uniform_grid.h>
#include <algorithm>

namespace chrray {
uniform_grid::uniform_grid(
    const std::vector<std::shared_ptr<hittable>>& objects,
    const aabb& world_box,
    int nx,
    int ny,
    int nz)
    : nx_(nx), ny_(ny), nz_(nz), box_(world_box) {
    cells_.resize(nx * ny * nz);
    step_ = euclidean_coordinate(
        (box_.max.x() - box_.min.x()) / nx, (box_.max.y() - box_.min.y()) / ny,
        (box_.max.z() - box_.min.z()) / nz);
    inv_step_ = euclidean_coordinate(
        1.0f / step_.x(), 1.0f / step_.y(), 1.0f / step_.z());
    for (const auto& obj : objects) {
        aabb b = obj->bounding_box();
        int ix0 =
            std::max(0, (int)((b.min.x() - box_.min.x()) * inv_step_.x()));
        int ix1 =
            std::min(nx - 1, (int)((b.max.x() - box_.min.x()) * inv_step_.x()));
        int iy0 =
            std::max(0, (int)((b.min.y() - box_.min.y()) * inv_step_.y()));
        int iy1 =
            std::min(ny - 1, (int)((b.max.y() - box_.min.y()) * inv_step_.y()));
        int iz0 =
            std::max(0, (int)((b.min.z() - box_.min.z()) * inv_step_.z()));
        int iz1 =
            std::min(nz - 1, (int)((b.max.z() - box_.min.z()) * inv_step_.z()));
        for (int ix = ix0; ix <= ix1; ++ix)
            for (int iy = iy0; iy <= iy1; ++iy)
                for (int iz = iz0; iz <= iz1; ++iz) {
                    cells_[ix + nx * (iy + ny * iz)].push_back(obj);
                }
    }
}
bool uniform_grid::intersect(
    const ray& r, float t_min, float t_max, hit_record& rec) const {
    float tx0, tx1, ty0, ty1, tz0, tz1;
    if (!box_.hit_interval(r.origin(), r.direction(), t_min, t_max, tx0, tx1))
        return false;
    int ix = (int)((r.at(tx0).x() - box_.min.x()) * inv_step_.x());
    int iy = (int)((r.at(tx0).y() - box_.min.y()) * inv_step_.y());
    int iz = (int)((r.at(tx0).z() - box_.min.z()) * inv_step_.z());
    ix = std::clamp(ix, 0, nx_ - 1);
    iy = std::clamp(iy, 0, ny_ - 1);
    iz = std::clamp(iz, 0, nz_ - 1);

    float t = tx0;
    bool hit = false;
    while (t <= tx1) {
        for (const auto& obj : cells_[ix + nx_ * (iy + ny_ * iz)]) {
            if (obj->intersect(r, t_min, t_max, rec)) {
                if (rec.t < tx1) {
                    hit = true;
                    t_max = rec.t;
                }
            }
        }
        if (hit) return true;
        float next_tx =
            (ix + (r.direction().x() > 0 ? 1 : 0)) * step_.x() + box_.min.x();
        float next_ty =
            (iy + (r.direction().y() > 0 ? 1 : 0)) * step_.y() + box_.min.y();
        float next_tz =
            (iz + (r.direction().z() > 0 ? 1 : 0)) * step_.z() + box_.min.z();
        float tx = (r.direction().x() != 0.0f)
                       ? (next_tx - r.origin().x()) / r.direction().x()
                       : inf;
        float ty = (r.direction().y() != 0.0f)
                       ? (next_ty - r.origin().y()) / r.direction().y()
                       : inf;
        float tz = (r.direction().z() != 0.0f)
                       ? (next_tz - r.origin().z()) / r.direction().z()
                       : inf;
        if (tx < ty && tx < tz) {
            ix += (r.direction().x() > 0 ? 1 : -1);
            t = tx;
        } else if (ty < tz) {
            iy += (r.direction().y() > 0 ? 1 : -1);
            t = ty;
        } else {
            iz += (r.direction().z() > 0 ? 1 : -1);
            t = tz;
        }
        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_)
            break;
    }
    return false;
}
bool uniform_grid::any_hit(const ray& r, float t_min, float t_max) const {
    float tx0, tx1;
    if (!box_.hit_interval(r.origin(), r.direction(), t_min, t_max, tx0, tx1))
        return false;

    int ix = (int)((r.at(tx0).x() - box_.min.x()) * inv_step_.x());
    int iy = (int)((r.at(tx0).y() - box_.min.y()) * inv_step_.y());
    int iz = (int)((r.at(tx0).z() - box_.min.z()) * inv_step_.z());
    ix = std::clamp(ix, 0, nx_ - 1);
    iy = std::clamp(iy, 0, ny_ - 1);
    iz = std::clamp(iz, 0, nz_ - 1);

    float t = tx0;

    while (t <= tx1) {
        for (const auto& obj : cells_[ix + nx_ * (iy + ny_ * iz)]) {
            if (obj->any_hit(r, t_min, t_max)) return true;
        }

        float next_tx =
            (ix + (r.direction().x() > 0 ? 1 : 0)) * step_.x() + box_.min.x();
        float next_ty =
            (iy + (r.direction().y() > 0 ? 1 : 0)) * step_.y() + box_.min.y();
        float next_tz =
            (iz + (r.direction().z() > 0 ? 1 : 0)) * step_.z() + box_.min.z();

        float tx = (r.direction().x() != 0.0f)
                       ? (next_tx - r.origin().x()) / r.direction().x()
                       : inf;
        float ty = (r.direction().y() != 0.0f)
                       ? (next_ty - r.origin().y()) / r.direction().y()
                       : inf;
        float tz = (r.direction().z() != 0.0f)
                       ? (next_tz - r.origin().z()) / r.direction().z()
                       : inf;

        if (tx < ty && tx < tz) {
            ix += (r.direction().x() > 0 ? 1 : -1);
            t = tx;
        } else if (ty < tz) {
            iy += (r.direction().y() > 0 ? 1 : -1);
            t = ty;
        } else {
            iz += (r.direction().z() > 0 ? 1 : -1);
            t = tz;
        }

        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_)
            break;
    }
    return false;
}
aabb uniform_grid::bounding_box() const {
    return box_;
}
}  // namespace chrray