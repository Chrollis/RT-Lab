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

ray8::ray8(const ray* rays, const __m256i& mask) : active(mask) {
    alignas(32) float ox[8], oy[8], oz[8];
    alignas(32) float dx[8], dy[8], dz[8];
    alignas(32) float tmin[8], tmax[8];

    for (int i = 0; i < 8; ++i) {
        int bit = 1 << i;
        if (_mm256_movemask_epi8(
                _mm256_and_si256(mask, _mm256_set1_epi32(bit))) != 0) {
            ox[i] = rays[i].origin().x();
            oy[i] = rays[i].origin().y();
            oz[i] = rays[i].origin().z();
            dx[i] = rays[i].direction().x();
            dy[i] = rays[i].direction().y();
            dz[i] = rays[i].direction().z();
            tmin[i] = 0.0f;
            tmax[i] = INFINITY;
        } else {
            ox[i] = oy[i] = oz[i] = dx[i] = dy[i] = dz[i] = 0.0f;
            tmin[i] = 0.0f;
            tmax[i] = -1.0f;
        }
    }
    this->ox = _mm256_load_ps(ox);
    this->oy = _mm256_load_ps(oy);
    this->oz = _mm256_load_ps(oz);
    this->dx = _mm256_load_ps(dx);
    this->dy = _mm256_load_ps(dy);
    this->dz = _mm256_load_ps(dz);
    this->tmin = _mm256_load_ps(tmin);
    this->tmax = _mm256_load_ps(tmax);
}
void hittable::intersect8(const ray8& rays, hit_record8& recs) const {
    recs.clear();
    alignas(32) float t_vals[8], px_vals[8], py_vals[8], pz_vals[8];
    alignas(32) float nx_vals[8], ny_vals[8], nz_vals[8];
    alignas(32) float u_vals[8], v_vals[8];
    int mat_ids[8] = {0};

    alignas(32) float ox_arr[8], oy_arr[8], oz_arr[8];
    alignas(32) float dx_arr[8], dy_arr[8], dz_arr[8];
    alignas(32) float tmin_arr[8], tmax_arr[8];
    _mm256_store_ps(ox_arr, rays.ox);
    _mm256_store_ps(oy_arr, rays.oy);
    _mm256_store_ps(oz_arr, rays.oz);
    _mm256_store_ps(dx_arr, rays.dx);
    _mm256_store_ps(dy_arr, rays.dy);
    _mm256_store_ps(dz_arr, rays.dz);
    _mm256_store_ps(tmin_arr, rays.tmin);
    _mm256_store_ps(tmax_arr, rays.tmax);

    __m256i active = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        if (_mm256_movemask_epi8(
                _mm256_and_si256(rays.active, _mm256_set1_epi32(1 << i))) == 0)
            continue;

        ray r(
            euclidean_coordinate(ox_arr[i], oy_arr[i], oz_arr[i]),
            euclidean_coordinate(dx_arr[i], dy_arr[i], dz_arr[i]));
        hit_record rec;
        if (intersect(r, tmin_arr[i], tmax_arr[i], rec)) {
            t_vals[i] = rec.t;
            px_vals[i] = rec.p.x();
            py_vals[i] = rec.p.y();
            pz_vals[i] = rec.p.z();
            nx_vals[i] = rec.n.x();
            ny_vals[i] = rec.n.y();
            nz_vals[i] = rec.n.z();
            u_vals[i] = rec.u;
            v_vals[i] = rec.v;
            mat_ids[i] = 0;
            active = _mm256_or_si256(active, _mm256_set1_epi32(1 << i));
        }
    }

    recs.t = _mm256_load_ps(t_vals);
    recs.px = _mm256_load_ps(px_vals);
    recs.py = _mm256_load_ps(py_vals);
    recs.pz = _mm256_load_ps(pz_vals);
    recs.nx = _mm256_load_ps(nx_vals);
    recs.ny = _mm256_load_ps(ny_vals);
    recs.nz = _mm256_load_ps(nz_vals);
    recs.u = _mm256_load_ps(u_vals);
    recs.v = _mm256_load_ps(v_vals);
    for (int i = 0; i < 8; ++i) recs.mat_id[i] = mat_ids[i];
    recs.active = active;
}

__m256i hittable::any_hit8(const ray8& rays) const {
    alignas(32) float ox_arr[8], oy_arr[8], oz_arr[8];
    alignas(32) float dx_arr[8], dy_arr[8], dz_arr[8];
    alignas(32) float tmin_arr[8], tmax_arr[8];
    _mm256_store_ps(ox_arr, rays.ox);
    _mm256_store_ps(oy_arr, rays.oy);
    _mm256_store_ps(oz_arr, rays.oz);
    _mm256_store_ps(dx_arr, rays.dx);
    _mm256_store_ps(dy_arr, rays.dy);
    _mm256_store_ps(dz_arr, rays.dz);
    _mm256_store_ps(tmin_arr, rays.tmin);
    _mm256_store_ps(tmax_arr, rays.tmax);

    __m256i result = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        if (_mm256_movemask_epi8(
                _mm256_and_si256(rays.active, _mm256_set1_epi32(1 << i))) == 0)
            continue;
        ray r(
            euclidean_coordinate(ox_arr[i], oy_arr[i], oz_arr[i]),
            euclidean_coordinate(dx_arr[i], dy_arr[i], dz_arr[i]));
        if (any_hit(r, tmin_arr[i], tmax_arr[i])) {
            result = _mm256_or_si256(result, _mm256_set1_epi32(1 << i));
        }
    }
    return result;
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
void sphere::intersect8(const ray8& rays, hit_record8& recs) const {
    __m256 cx = _mm256_set1_ps(center_.x());
    __m256 cy = _mm256_set1_ps(center_.y());
    __m256 cz = _mm256_set1_ps(center_.z());
    __m256 r = _mm256_set1_ps(radius_);

    __m256 ocx = _mm256_sub_ps(rays.ox, cx);
    __m256 ocy = _mm256_sub_ps(rays.oy, cy);
    __m256 ocz = _mm256_sub_ps(rays.oz, cz);

    __m256 a = _mm256_fmadd_ps(
        rays.dx, rays.dx,
        _mm256_fmadd_ps(rays.dy, rays.dy, _mm256_mul_ps(rays.dz, rays.dz)));

    __m256 b = _mm256_mul_ps(
        _mm256_set1_ps(2.0f),
        _mm256_fmadd_ps(
            ocx, rays.dx,
            _mm256_fmadd_ps(ocy, rays.dy, _mm256_mul_ps(ocz, rays.dz))));

    __m256 c = _mm256_sub_ps(
        _mm256_fmadd_ps(
            ocx, ocx, _mm256_fmadd_ps(ocy, ocy, _mm256_mul_ps(ocz, ocz))),
        _mm256_mul_ps(r, r));

    __m256 disc = _mm256_sub_ps(
        _mm256_mul_ps(b, b),
        _mm256_mul_ps(_mm256_set1_ps(4.0f), _mm256_mul_ps(a, c)));

    __m256 disc_pos = _mm256_max_ps(disc, _mm256_setzero_ps());
    __m256 sqrt_disc = _mm256_sqrt_ps(disc_pos);

    __m256 two_a = _mm256_mul_ps(_mm256_set1_ps(2.0f), a);
    __m256 neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);
    __m256 t1 = _mm256_div_ps(_mm256_sub_ps(neg_b, sqrt_disc), two_a);
    __m256 t2 = _mm256_div_ps(_mm256_add_ps(neg_b, sqrt_disc), two_a);

    __m256 tmin_vec = rays.tmin;
    __m256 tmax_vec = rays.tmax;
    __m256 valid1 = _mm256_and_ps(
        _mm256_cmp_ps(t1, tmin_vec, _CMP_GE_OQ),
        _mm256_cmp_ps(t1, tmax_vec, _CMP_LE_OQ));
    __m256 valid2 = _mm256_and_ps(
        _mm256_cmp_ps(t2, tmin_vec, _CMP_GE_OQ),
        _mm256_cmp_ps(t2, tmax_vec, _CMP_LE_OQ));
    __m256 valid = _mm256_or_ps(valid1, valid2);

    __m256 t_hit = _mm256_blendv_ps(t2, t1, valid1);

    __m256 px = _mm256_fmadd_ps(t_hit, rays.dx, rays.ox);
    __m256 py = _mm256_fmadd_ps(t_hit, rays.dy, rays.oy);
    __m256 pz = _mm256_fmadd_ps(t_hit, rays.dz, rays.oz);

    __m256 nx = _mm256_div_ps(_mm256_sub_ps(px, cx), r);
    __m256 ny = _mm256_div_ps(_mm256_sub_ps(py, cy), r);
    __m256 nz = _mm256_div_ps(_mm256_sub_ps(pz, cz), r);

    __m256 dot = _mm256_fmadd_ps(
        nx, rays.dx, _mm256_fmadd_ps(ny, rays.dy, _mm256_mul_ps(nz, rays.dz)));
    __m256 flip = _mm256_cmp_ps(dot, _mm256_setzero_ps(), _CMP_GT_OQ);
    nx = _mm256_blendv_ps(nx, _mm256_sub_ps(_mm256_setzero_ps(), nx), flip);
    ny = _mm256_blendv_ps(ny, _mm256_sub_ps(_mm256_setzero_ps(), ny), flip);
    nz = _mm256_blendv_ps(nz, _mm256_sub_ps(_mm256_setzero_ps(), nz), flip);

    __m256 zero = _mm256_setzero_ps();

    recs.t = t_hit;
    recs.px = px;
    recs.py = py;
    recs.pz = pz;
    recs.nx = nx;
    recs.ny = ny;
    recs.nz = nz;
    recs.u = zero;
    recs.v = zero;
    for (int i = 0; i < 8; ++i) recs.mat_id[i] = mat_index_;
    recs.active = _mm256_castps_si256(valid);
}

__m256i sphere::any_hit8(const ray8& rays) const {
    __m256 cx = _mm256_set1_ps(center_.x());
    __m256 cy = _mm256_set1_ps(center_.y());
    __m256 cz = _mm256_set1_ps(center_.z());
    __m256 r = _mm256_set1_ps(radius_);

    __m256 ocx = _mm256_sub_ps(rays.ox, cx);
    __m256 ocy = _mm256_sub_ps(rays.oy, cy);
    __m256 ocz = _mm256_sub_ps(rays.oz, cz);

    __m256 a = _mm256_fmadd_ps(
        rays.dx, rays.dx,
        _mm256_fmadd_ps(rays.dy, rays.dy, _mm256_mul_ps(rays.dz, rays.dz)));
    __m256 b = _mm256_mul_ps(
        _mm256_set1_ps(2.0f),
        _mm256_fmadd_ps(
            ocx, rays.dx,
            _mm256_fmadd_ps(ocy, rays.dy, _mm256_mul_ps(ocz, rays.dz))));
    __m256 c = _mm256_sub_ps(
        _mm256_fmadd_ps(
            ocx, ocx, _mm256_fmadd_ps(ocy, ocy, _mm256_mul_ps(ocz, ocz))),
        _mm256_mul_ps(r, r));

    __m256 disc = _mm256_sub_ps(
        _mm256_mul_ps(b, b),
        _mm256_mul_ps(_mm256_set1_ps(4.0f), _mm256_mul_ps(a, c)));
    __m256 disc_pos = _mm256_max_ps(disc, _mm256_setzero_ps());
    __m256 sqrt_disc = _mm256_sqrt_ps(disc_pos);
    __m256 two_a = _mm256_mul_ps(_mm256_set1_ps(2.0f), a);
    __m256 neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);
    __m256 t1 = _mm256_div_ps(_mm256_sub_ps(neg_b, sqrt_disc), two_a);
    __m256 t2 = _mm256_div_ps(_mm256_add_ps(neg_b, sqrt_disc), two_a);

    __m256 valid1 = _mm256_and_ps(
        _mm256_cmp_ps(t1, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t1, rays.tmax, _CMP_LE_OQ));
    __m256 valid2 = _mm256_and_ps(
        _mm256_cmp_ps(t2, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t2, rays.tmax, _CMP_LE_OQ));
    __m256 valid = _mm256_or_ps(valid1, valid2);

    return _mm256_castps_si256(valid);
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
void triangle::intersect8(const ray8& rays, hit_record8& recs) const {
    __m256 v0x = _mm256_set1_ps(tri_.v0.x());
    __m256 v0y = _mm256_set1_ps(tri_.v0.y());
    __m256 v0z = _mm256_set1_ps(tri_.v0.z());
    __m256 v1x = _mm256_set1_ps(tri_.v1.x());
    __m256 v1y = _mm256_set1_ps(tri_.v1.y());
    __m256 v1z = _mm256_set1_ps(tri_.v1.z());
    __m256 v2x = _mm256_set1_ps(tri_.v2.x());
    __m256 v2y = _mm256_set1_ps(tri_.v2.y());
    __m256 v2z = _mm256_set1_ps(tri_.v2.z());

    __m256 e1x = _mm256_sub_ps(v1x, v0x);
    __m256 e1y = _mm256_sub_ps(v1y, v0y);
    __m256 e1z = _mm256_sub_ps(v1z, v0z);
    __m256 e2x = _mm256_sub_ps(v2x, v0x);
    __m256 e2y = _mm256_sub_ps(v2y, v0y);
    __m256 e2z = _mm256_sub_ps(v2z, v0z);

    __m256 pvecx =
        _mm256_sub_ps(_mm256_mul_ps(rays.dy, e2z), _mm256_mul_ps(rays.dz, e2y));
    __m256 pvecy =
        _mm256_sub_ps(_mm256_mul_ps(rays.dz, e2x), _mm256_mul_ps(rays.dx, e2z));
    __m256 pvecz =
        _mm256_sub_ps(_mm256_mul_ps(rays.dx, e2y), _mm256_mul_ps(rays.dy, e2x));

    __m256 det = _mm256_fmadd_ps(
        e1x, pvecx, _mm256_fmadd_ps(e1y, pvecy, _mm256_mul_ps(e1z, pvecz)));

    __m256 abs_det =
        _mm256_and_ps(det, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 valid_det = _mm256_cmp_ps(abs_det, _mm256_set1_ps(eps), _CMP_GT_OQ);

    __m256 inv_det = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

    __m256 tvecx = _mm256_sub_ps(rays.ox, v0x);
    __m256 tvecy = _mm256_sub_ps(rays.oy, v0y);
    __m256 tvecz = _mm256_sub_ps(rays.oz, v0z);

    __m256 u = _mm256_mul_ps(
        _mm256_fmadd_ps(
            tvecx, pvecx,
            _mm256_fmadd_ps(tvecy, pvecy, _mm256_mul_ps(tvecz, pvecz))),
        inv_det);

    __m256 valid_u = _mm256_and_ps(
        _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GE_OQ),
        _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 qvecx =
        _mm256_sub_ps(_mm256_mul_ps(tvecy, e1z), _mm256_mul_ps(tvecz, e1y));
    __m256 qvecy =
        _mm256_sub_ps(_mm256_mul_ps(tvecz, e1x), _mm256_mul_ps(tvecx, e1z));
    __m256 qvecz =
        _mm256_sub_ps(_mm256_mul_ps(tvecx, e1y), _mm256_mul_ps(tvecy, e1x));

    __m256 v = _mm256_mul_ps(
        _mm256_fmadd_ps(
            rays.dx, qvecx,
            _mm256_fmadd_ps(rays.dy, qvecy, _mm256_mul_ps(rays.dz, qvecz))),
        inv_det);

    __m256 valid_v = _mm256_and_ps(
        _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ),
        _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 t = _mm256_mul_ps(
        _mm256_fmadd_ps(
            e2x, qvecx, _mm256_fmadd_ps(e2y, qvecy, _mm256_mul_ps(e2z, qvecz))),
        inv_det);

    __m256 valid_t = _mm256_and_ps(
        _mm256_cmp_ps(t, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t, rays.tmax, _CMP_LE_OQ));

    __m256 valid = _mm256_and_ps(
        valid_det, _mm256_and_ps(valid_u, _mm256_and_ps(valid_v, valid_t)));

    __m256 px = _mm256_fmadd_ps(t, rays.dx, rays.ox);
    __m256 py = _mm256_fmadd_ps(t, rays.dy, rays.oy);
    __m256 pz = _mm256_fmadd_ps(t, rays.dz, rays.oz);

    __m256 nx = _mm256_set1_ps(tri_.normal.x());
    __m256 ny = _mm256_set1_ps(tri_.normal.y());
    __m256 nz = _mm256_set1_ps(tri_.normal.z());

    __m256 dot_nd = _mm256_fmadd_ps(
        nx, rays.dx, _mm256_fmadd_ps(ny, rays.dy, _mm256_mul_ps(nz, rays.dz)));
    __m256 flip = _mm256_cmp_ps(dot_nd, _mm256_setzero_ps(), _CMP_GT_OQ);
    nx = _mm256_blendv_ps(nx, _mm256_sub_ps(_mm256_setzero_ps(), nx), flip);
    ny = _mm256_blendv_ps(ny, _mm256_sub_ps(_mm256_setzero_ps(), ny), flip);
    nz = _mm256_blendv_ps(nz, _mm256_sub_ps(_mm256_setzero_ps(), nz), flip);

    recs.t = t;
    recs.px = px;
    recs.py = py;
    recs.pz = pz;
    recs.nx = nx;
    recs.ny = ny;
    recs.nz = nz;
    recs.u = u;
    recs.v = v;
    for (int i = 0; i < 8; ++i) recs.mat_id[i] = mat_index_;
    recs.active = _mm256_castps_si256(valid);
}

__m256i triangle::any_hit8(const ray8& rays) const {
    __m256 v0x = _mm256_set1_ps(tri_.v0.x());
    __m256 v0y = _mm256_set1_ps(tri_.v0.y());
    __m256 v0z = _mm256_set1_ps(tri_.v0.z());
    __m256 v1x = _mm256_set1_ps(tri_.v1.x());
    __m256 v1y = _mm256_set1_ps(tri_.v1.y());
    __m256 v1z = _mm256_set1_ps(tri_.v1.z());
    __m256 v2x = _mm256_set1_ps(tri_.v2.x());
    __m256 v2y = _mm256_set1_ps(tri_.v2.y());
    __m256 v2z = _mm256_set1_ps(tri_.v2.z());

    __m256 e1x = _mm256_sub_ps(v1x, v0x);
    __m256 e1y = _mm256_sub_ps(v1y, v0y);
    __m256 e1z = _mm256_sub_ps(v1z, v0z);
    __m256 e2x = _mm256_sub_ps(v2x, v0x);
    __m256 e2y = _mm256_sub_ps(v2y, v0y);
    __m256 e2z = _mm256_sub_ps(v2z, v0z);

    __m256 pvecx =
        _mm256_sub_ps(_mm256_mul_ps(rays.dy, e2z), _mm256_mul_ps(rays.dz, e2y));
    __m256 pvecy =
        _mm256_sub_ps(_mm256_mul_ps(rays.dz, e2x), _mm256_mul_ps(rays.dx, e2z));
    __m256 pvecz =
        _mm256_sub_ps(_mm256_mul_ps(rays.dx, e2y), _mm256_mul_ps(rays.dy, e2x));

    __m256 det = _mm256_fmadd_ps(
        e1x, pvecx, _mm256_fmadd_ps(e1y, pvecy, _mm256_mul_ps(e1z, pvecz)));
    __m256 abs_det =
        _mm256_and_ps(det, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 valid_det = _mm256_cmp_ps(abs_det, _mm256_set1_ps(eps), _CMP_GT_OQ);
    __m256 inv_det = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

    __m256 tvecx = _mm256_sub_ps(rays.ox, v0x);
    __m256 tvecy = _mm256_sub_ps(rays.oy, v0y);
    __m256 tvecz = _mm256_sub_ps(rays.oz, v0z);

    __m256 u = _mm256_mul_ps(
        _mm256_fmadd_ps(
            tvecx, pvecx,
            _mm256_fmadd_ps(tvecy, pvecy, _mm256_mul_ps(tvecz, pvecz))),
        inv_det);
    __m256 valid_u = _mm256_and_ps(
        _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GE_OQ),
        _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 qvecx =
        _mm256_sub_ps(_mm256_mul_ps(tvecy, e1z), _mm256_mul_ps(tvecz, e1y));
    __m256 qvecy =
        _mm256_sub_ps(_mm256_mul_ps(tvecz, e1x), _mm256_mul_ps(tvecx, e1z));
    __m256 qvecz =
        _mm256_sub_ps(_mm256_mul_ps(tvecx, e1y), _mm256_mul_ps(tvecy, e1x));

    __m256 v = _mm256_mul_ps(
        _mm256_fmadd_ps(
            rays.dx, qvecx,
            _mm256_fmadd_ps(rays.dy, qvecy, _mm256_mul_ps(rays.dz, qvecz))),
        inv_det);
    __m256 valid_v = _mm256_and_ps(
        _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ),
        _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 t = _mm256_mul_ps(
        _mm256_fmadd_ps(
            e2x, qvecx, _mm256_fmadd_ps(e2y, qvecy, _mm256_mul_ps(e2z, qvecz))),
        inv_det);
    __m256 valid_t = _mm256_and_ps(
        _mm256_cmp_ps(t, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t, rays.tmax, _CMP_LE_OQ));

    __m256 valid = _mm256_and_ps(
        valid_det, _mm256_and_ps(valid_u, _mm256_and_ps(valid_v, valid_t)));
    return _mm256_castps_si256(valid);
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
void plane::intersect8(const ray8& rays, hit_record8& recs) const {
    __m256 nx = _mm256_set1_ps(normal_.x());
    __m256 ny = _mm256_set1_ps(normal_.y());
    __m256 nz = _mm256_set1_ps(normal_.z());
    __m256 px = _mm256_set1_ps(point_.x());
    __m256 py = _mm256_set1_ps(point_.y());
    __m256 pz = _mm256_set1_ps(point_.z());

    __m256 denom = _mm256_fmadd_ps(
        rays.dx, nx, _mm256_fmadd_ps(rays.dy, ny, _mm256_mul_ps(rays.dz, nz)));
    __m256 abs_denom = _mm256_and_ps(
        denom, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 valid_denom =
        _mm256_cmp_ps(abs_denom, _mm256_set1_ps(eps), _CMP_GT_OQ);

    __m256 t = _mm256_div_ps(
        _mm256_fmadd_ps(
            _mm256_sub_ps(px, rays.ox), nx,
            _mm256_fmadd_ps(
                _mm256_sub_ps(py, rays.oy), ny,
                _mm256_mul_ps(_mm256_sub_ps(pz, rays.oz), nz))),
        denom);

    __m256 valid_t = _mm256_and_ps(
        _mm256_cmp_ps(t, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t, rays.tmax, _CMP_LE_OQ));
    __m256 valid = _mm256_and_ps(valid_denom, valid_t);

    __m256 hitx = _mm256_fmadd_ps(t, rays.dx, rays.ox);
    __m256 hity = _mm256_fmadd_ps(t, rays.dy, rays.oy);
    __m256 hitz = _mm256_fmadd_ps(t, rays.dz, rays.oz);

    __m256 dot_nd = _mm256_fmadd_ps(
        nx, rays.dx, _mm256_fmadd_ps(ny, rays.dy, _mm256_mul_ps(nz, rays.dz)));
    __m256 flip = _mm256_cmp_ps(dot_nd, _mm256_setzero_ps(), _CMP_GT_OQ);
    __m256 res_nx =
        _mm256_blendv_ps(nx, _mm256_sub_ps(_mm256_setzero_ps(), nx), flip);
    __m256 res_ny =
        _mm256_blendv_ps(ny, _mm256_sub_ps(_mm256_setzero_ps(), ny), flip);
    __m256 res_nz =
        _mm256_blendv_ps(nz, _mm256_sub_ps(_mm256_setzero_ps(), nz), flip);

    recs.t = t;
    recs.px = hitx;
    recs.py = hity;
    recs.pz = hitz;
    recs.nx = res_nx;
    recs.ny = res_ny;
    recs.nz = res_nz;
    recs.u = _mm256_setzero_ps();
    recs.v = _mm256_setzero_ps();
    for (int i = 0; i < 8; ++i) recs.mat_id[i] = mat_index_;
    recs.active = _mm256_castps_si256(valid);
}

__m256i plane::any_hit8(const ray8& rays) const {
    __m256 nx = _mm256_set1_ps(normal_.x());
    __m256 ny = _mm256_set1_ps(normal_.y());
    __m256 nz = _mm256_set1_ps(normal_.z());
    __m256 px = _mm256_set1_ps(point_.x());
    __m256 py = _mm256_set1_ps(point_.y());
    __m256 pz = _mm256_set1_ps(point_.z());

    __m256 denom = _mm256_fmadd_ps(
        rays.dx, nx, _mm256_fmadd_ps(rays.dy, ny, _mm256_mul_ps(rays.dz, nz)));
    __m256 abs_denom = _mm256_and_ps(
        denom, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 valid_denom =
        _mm256_cmp_ps(abs_denom, _mm256_set1_ps(eps), _CMP_GT_OQ);
    __m256 t = _mm256_div_ps(
        _mm256_fmadd_ps(
            _mm256_sub_ps(px, rays.ox), nx,
            _mm256_fmadd_ps(
                _mm256_sub_ps(py, rays.oy), ny,
                _mm256_mul_ps(_mm256_sub_ps(pz, rays.oz), nz))),
        denom);
    __m256 valid_t = _mm256_and_ps(
        _mm256_cmp_ps(t, rays.tmin, _CMP_GE_OQ),
        _mm256_cmp_ps(t, rays.tmax, _CMP_LE_OQ));
    __m256 valid = _mm256_and_ps(valid_denom, valid_t);
    return _mm256_castps_si256(valid);
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
void mesh::intersect8(const ray8& rays, hit_record8& recs) const {
    recs.clear();
    if (triangles_.empty()) return;

    alignas(32) float best_t[8];
    alignas(32) float best_px[8], best_py[8], best_pz[8];
    alignas(32) float best_nx[8], best_ny[8], best_nz[8];
    alignas(32) float best_u[8], best_v[8];
    int best_mat_id[8];
    __m256i best_active = _mm256_setzero_si256();

    for (int i = 0; i < 8; ++i) best_t[i] = INFINITY;

    for (const auto& tri : triangles_) {
        hit_record8 temp;
        tri->intersect8(rays, temp);

        alignas(32) float temp_t[8], temp_px[8], temp_py[8], temp_pz[8];
        alignas(32) float temp_nx[8], temp_ny[8], temp_nz[8];
        alignas(32) float temp_u[8], temp_v[8];
        int temp_mat_id[8];
        alignas(32) int temp_active[8];

        _mm256_store_ps(temp_t, temp.t);
        _mm256_store_ps(temp_px, temp.px);
        _mm256_store_ps(temp_py, temp.py);
        _mm256_store_ps(temp_pz, temp.pz);
        _mm256_store_ps(temp_nx, temp.nx);
        _mm256_store_ps(temp_ny, temp.ny);
        _mm256_store_ps(temp_nz, temp.nz);
        _mm256_store_ps(temp_u, temp.u);
        _mm256_store_ps(temp_v, temp.v);
        for (int i = 0; i < 8; ++i) temp_mat_id[i] = temp.mat_id[i];
        _mm256_store_si256((__m256i*)temp_active, temp.active);

        for (int i = 0; i < 8; ++i) {
            if (temp_active[i] && temp_t[i] < best_t[i]) {
                best_t[i] = temp_t[i];
                best_px[i] = temp_px[i];
                best_py[i] = temp_py[i];
                best_pz[i] = temp_pz[i];
                best_nx[i] = temp_nx[i];
                best_ny[i] = temp_ny[i];
                best_nz[i] = temp_nz[i];
                best_u[i] = temp_u[i];
                best_v[i] = temp_v[i];
                best_mat_id[i] = temp_mat_id[i];

                best_active =
                    _mm256_or_si256(best_active, _mm256_set1_epi32(1 << i));
            }
        }
    }

    recs.t = _mm256_load_ps(best_t);
    recs.px = _mm256_load_ps(best_px);
    recs.py = _mm256_load_ps(best_py);
    recs.pz = _mm256_load_ps(best_pz);
    recs.nx = _mm256_load_ps(best_nx);
    recs.ny = _mm256_load_ps(best_ny);
    recs.nz = _mm256_load_ps(best_nz);
    recs.u = _mm256_load_ps(best_u);
    recs.v = _mm256_load_ps(best_v);
    for (int i = 0; i < 8; ++i) recs.mat_id[i] = best_mat_id[i];
    recs.active = best_active;
}

__m256i mesh::any_hit8(const ray8& rays) const {
    __m256i result = _mm256_setzero_si256();
    for (const auto& tri : triangles_) {
        result = _mm256_or_si256(result, tri->any_hit8(rays));
        if (_mm256_movemask_epi8(result) == 0xFF) break;
    }
    return result;
}
}  // namespace chrray