#include <matrix.h>
#include <cmath>
#include <stdexcept>

namespace chrray {
static inline __m128 broadcast_scalar(float s) {
    return _mm_set_ps1(s);
}
static inline float horizontal_sum(__m128 v) {
    __m128 t1 = _mm_hadd_ps(v, v);
    __m128 t2 = _mm_hadd_ps(t1, t1);
    return _mm_cvtss_f32(t2);
}

euclidean_coordinate::euclidean_coordinate() : data_(_mm_setzero_ps()) {}
euclidean_coordinate::euclidean_coordinate(
    std::initializer_list<float> values) {
    if (values.size() != 3) {
        throw std::invalid_argument(
            "Euclidean coordinate must have 3 components");
    }
    float arr[4] = {0, 0, 0, 0};
    std::copy(values.begin(), values.end(), arr);
    data_ = _mm_loadu_ps(arr);
}
euclidean_coordinate::euclidean_coordinate(float x, float y, float z)
    : data_(_mm_set_ps(0.0f, z, y, x)) {}
euclidean_coordinate::euclidean_coordinate(__m128 data) : data_(data) {}
float euclidean_coordinate::x() const {
    return _mm_cvtss_f32(data_);
}
float euclidean_coordinate::y() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(1, 1, 1, 1)));
}
float euclidean_coordinate::z() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 2, 2, 2)));
}
euclidean_coordinate euclidean_coordinate::operator+(
    const euclidean_coordinate& other) const {
    return euclidean_coordinate(_mm_add_ps(data_, other.data_));
}
euclidean_coordinate euclidean_coordinate::operator-(
    const euclidean_coordinate& other) const {
    return euclidean_coordinate(_mm_sub_ps(data_, other.data_));
}
euclidean_coordinate euclidean_coordinate::operator*(float scalar) const {
    return euclidean_coordinate(_mm_mul_ps(data_, broadcast_scalar(scalar)));
}
euclidean_coordinate euclidean_coordinate::operator/(float scalar) const {
    if (scalar == 0.0f) throw std::invalid_argument("Division by zero");
    return euclidean_coordinate(_mm_div_ps(data_, broadcast_scalar(scalar)));
}
float euclidean_coordinate::dot(const euclidean_coordinate& other) const {
    __m128 mul = _mm_mul_ps(data_, other.data_);
    return horizontal_sum(mul);
}
euclidean_coordinate euclidean_coordinate::cross(
    const euclidean_coordinate& other) const {
    __m128 a = data_, b = other.data_;
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    return euclidean_coordinate(
        _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx)));
}
float euclidean_coordinate::length() const {
    return std::sqrtf(dot(*this));
}
euclidean_coordinate euclidean_coordinate::normalize() const {
    float len = length();
    if (len == 0.0f) {
        throw std::invalid_argument("Cannot normalize a zero-length vector");
    }
    return *this / len;
}
homogeneous_coordinate euclidean_coordinate::homogenize() const {
    return homogeneous_coordinate(x(), y(), z(), 1.0f);
}
euclidean_coordinate euclidean_coordinate::operator-() const {
    return euclidean_coordinate(_mm_sub_ps(_mm_setzero_ps(), data_));
}
euclidean_coordinate operator*(float scalar, const euclidean_coordinate& vec) {
    return vec * scalar;
}
bool euclidean_coordinate::operator==(const euclidean_coordinate& other) const {
    __m128 diff = _mm_sub_ps(data_, other.data_);
    return _mm_movemask_ps(_mm_cmpneq_ps(diff, _mm_setzero_ps())) == 0;
}
euclidean_coordinate euclidean_coordinate::with_x(float new_x) const {
    return euclidean_coordinate(new_x, y(), z());
}
euclidean_coordinate euclidean_coordinate::with_y(float new_y) const {
    return euclidean_coordinate(x(), new_y, z());
}
euclidean_coordinate euclidean_coordinate::with_z(float new_z) const {
    return euclidean_coordinate(x(), y(), new_z);
}

homogeneous_coordinate::homogeneous_coordinate() : data_(_mm_setzero_ps()) {}
homogeneous_coordinate::homogeneous_coordinate(
    std::initializer_list<float> values) {
    if (values.size() != 4)
        throw std::invalid_argument(
            "Homogeneous coordinate must have 4 components");
    float arr[4];
    std::copy(values.begin(), values.end(), arr);
    data_ = _mm_loadu_ps(arr);
}
homogeneous_coordinate::homogeneous_coordinate(
    float x, float y, float z, float w)
    : data_(_mm_set_ps(w, z, y, x)) {}
homogeneous_coordinate::homogeneous_coordinate(__m128 data) : data_(data) {}

float homogeneous_coordinate::x() const {
    return _mm_cvtss_f32(data_);
}
float homogeneous_coordinate::y() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(1, 1, 1, 1)));
}
float homogeneous_coordinate::z() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 2, 2, 2)));
}
float homogeneous_coordinate::w() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(3, 3, 3, 3)));
}

homogeneous_coordinate homogeneous_coordinate::operator+(
    const homogeneous_coordinate& other) const {
    return homogeneous_coordinate(_mm_add_ps(data_, other.data_));
}
homogeneous_coordinate homogeneous_coordinate::operator-(
    const homogeneous_coordinate& other) const {
    return homogeneous_coordinate(_mm_sub_ps(data_, other.data_));
}
homogeneous_coordinate homogeneous_coordinate::operator*(float scalar) const {
    return homogeneous_coordinate(_mm_mul_ps(data_, broadcast_scalar(scalar)));
}
homogeneous_coordinate homogeneous_coordinate::operator/(float scalar) const {
    if (scalar == 0.0f) throw std::invalid_argument("Division by zero");
    return homogeneous_coordinate(_mm_div_ps(data_, broadcast_scalar(scalar)));
}
float homogeneous_coordinate::dot(const homogeneous_coordinate& other) const {
    __m128 mul = _mm_mul_ps(data_, other.data_);
    return horizontal_sum(mul);
}
float homogeneous_coordinate::length() const {
    return std::sqrtf(dot(*this));
}
homogeneous_coordinate homogeneous_coordinate::normalize() const {
    float len = length();
    if (len == 0.0f)
        throw std::invalid_argument("Cannot normalize zero-length vector");
    return *this / len;
}
euclidean_coordinate homogeneous_coordinate::dehomogenize() const {
    float wv = w();
    if (wv == 0.0f)
        throw std::invalid_argument("Cannot dehomogenize point with w=0");
    return euclidean_coordinate(x() / wv, y() / wv, z() / wv);
}
homogeneous_coordinate homogeneous_coordinate::operator-() const {
    return homogeneous_coordinate(_mm_sub_ps(_mm_setzero_ps(), data_));
}
homogeneous_coordinate operator*(
    float scalar, const homogeneous_coordinate& vec) {
    return vec * scalar;
}
bool homogeneous_coordinate::operator==(
    const homogeneous_coordinate& other) const {
    __m128 diff = _mm_sub_ps(data_, other.data_);
    return _mm_movemask_ps(_mm_cmpneq_ps(diff, _mm_setzero_ps())) == 0;
}
homogeneous_coordinate homogeneous_coordinate::with_x(float new_x) const {
    return homogeneous_coordinate(new_x, y(), z(), w());
}
homogeneous_coordinate homogeneous_coordinate::with_y(float new_y) const {
    return homogeneous_coordinate(x(), new_y, z(), w());
}
homogeneous_coordinate homogeneous_coordinate::with_z(float new_z) const {
    return homogeneous_coordinate(x(), y(), new_z, w());
}
homogeneous_coordinate homogeneous_coordinate::with_w(float new_w) const {
    return homogeneous_coordinate(x(), y(), z(), new_w);
}

quaternion::quaternion() : data_(_mm_setzero_ps()) {}
quaternion::quaternion(float x, float y, float z, float w)
    : data_(_mm_set_ps(w, z, y, x)) {}
quaternion::quaternion(std::initializer_list<float> values) {
    if (values.size() != 4)
        throw std::invalid_argument("Quaternion must have 4 components");
    float arr[4];
    std::copy(values.begin(), values.end(), arr);
    data_ = _mm_loadu_ps(arr);
}
quaternion::quaternion(__m128 data) : data_(data) {}

float quaternion::x() const {
    return _mm_cvtss_f32(data_);
}
float quaternion::y() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(1, 1, 1, 1)));
}
float quaternion::z() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(2, 2, 2, 2)));
}
float quaternion::w() const {
    return _mm_cvtss_f32(_mm_shuffle_ps(data_, data_, _MM_SHUFFLE(3, 3, 3, 3)));
}

quaternion quaternion::operator*(const quaternion& other) const {
    float x1 = x(), y1 = y(), z1 = z(), w1 = w();
    float x2 = other.x(), y2 = other.y(), z2 = other.z(), w2 = other.w();
    return quaternion(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
}
quaternion quaternion::conjugate() const {
    return quaternion(-x(), -y(), -z(), w());
}
float quaternion::length() const {
    __m128 sq = _mm_mul_ps(data_, data_);
    return std::sqrtf(horizontal_sum(sq));
}
quaternion quaternion::normalize() const {
    float len = length();
    if (len == 0.0f)
        throw std::invalid_argument("Cannot normalize zero-length quaternion");
    float inv_len = 1.0f / len;
    return quaternion(_mm_mul_ps(data_, broadcast_scalar(inv_len)));
}
euclidean_coordinate quaternion::rotate_vector(
    const euclidean_coordinate& vec) const {
    quaternion q = normalize();
    quaternion v(vec.x(), vec.y(), vec.z(), 0.0f);
    quaternion rotated = q * v * q.conjugate();
    return euclidean_coordinate(rotated.x(), rotated.y(), rotated.z());
}
quaternion quaternion::from_axis_angle(
    const euclidean_coordinate& axis, float angle) {
    euclidean_coordinate na = axis.normalize();
    float half = angle * 0.5f;
    float s = std::sinf(half);
    return quaternion(na.x() * s, na.y() * s, na.z() * s, std::cosf(half));
}
bool quaternion::operator==(const quaternion& other) const {
    __m128 diff = _mm_sub_ps(data_, other.data_);
    return _mm_movemask_ps(_mm_cmpneq_ps(diff, _mm_setzero_ps())) == 0;
}
quaternion quaternion::with_x(float new_x) const {
    return quaternion(new_x, y(), z(), w());
}
quaternion quaternion::with_y(float new_y) const {
    return quaternion(x(), new_y, z(), w());
}
quaternion quaternion::with_z(float new_z) const {
    return quaternion(x(), y(), new_z, w());
}
quaternion quaternion::with_w(float new_w) const {
    return quaternion(x(), y(), z(), new_w);
}

homogeneous_transform::homogeneous_transform() {
    rows_[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    rows_[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    rows_[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    rows_[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
}
homogeneous_transform::homogeneous_transform(
    std::initializer_list<std::initializer_list<float>> values) {
    if (values.size() != 4)
        throw std::invalid_argument("Homogeneous transform must be 4x4");
    int i = 0;
    for (const auto& row : values) {
        if (row.size() != 4)
            throw std::invalid_argument("Homogeneous transform must be 4x4");
        float arr[4];
        std::copy(row.begin(), row.end(), arr);
        rows_[i++] = _mm_loadu_ps(arr);
    }
}
homogeneous_transform::homogeneous_transform(const std::array<__m128, 4>& rows)
    : rows_(rows) {}

float homogeneous_transform::operator()(size_t row, size_t col) const {
    alignas(16) float arr[4];
    _mm_store_ps(arr, rows_[row]);
    return arr[col];
}
homogeneous_transform homogeneous_transform::operator*(
    const homogeneous_transform& other) const {
    __m128 b_col0 =
        _mm_set_ps(other(3, 0), other(2, 0), other(1, 0), other(0, 0));
    __m128 b_col1 =
        _mm_set_ps(other(3, 1), other(2, 1), other(1, 1), other(0, 1));
    __m128 b_col2 =
        _mm_set_ps(other(3, 2), other(2, 2), other(1, 2), other(0, 2));
    __m128 b_col3 =
        _mm_set_ps(other(3, 3), other(2, 3), other(1, 3), other(0, 3));

    std::array<__m128, 4> result_rows;
    for (int i = 0; i < 4; ++i) {
        __m128 a_row = rows_[i];
        __m128 c0 = _mm_mul_ps(a_row, b_col0);
        __m128 c1 = _mm_mul_ps(a_row, b_col1);
        __m128 c2 = _mm_mul_ps(a_row, b_col2);
        __m128 c3 = _mm_mul_ps(a_row, b_col3);
        float sum0 = horizontal_sum(c0);
        float sum1 = horizontal_sum(c1);
        float sum2 = horizontal_sum(c2);
        float sum3 = horizontal_sum(c3);
        result_rows[i] = _mm_set_ps(sum3, sum2, sum1, sum0);
    }
    return homogeneous_transform(result_rows);
}
homogeneous_coordinate homogeneous_transform::operator*(
    const homogeneous_coordinate& vec) const {
    __m128 v = _mm_set_ps(vec.w(), vec.z(), vec.y(), vec.x());
    __m128 res0 = _mm_mul_ps(rows_[0], v);
    __m128 res1 = _mm_mul_ps(rows_[1], v);
    __m128 res2 = _mm_mul_ps(rows_[2], v);
    __m128 res3 = _mm_mul_ps(rows_[3], v);
    float x = horizontal_sum(res0);
    float y = horizontal_sum(res1);
    float z = horizontal_sum(res2);
    float w = horizontal_sum(res3);
    return homogeneous_coordinate(x, y, z, w);
}
homogeneous_transform homogeneous_transform::translation(
    const euclidean_coordinate& vec) {
    homogeneous_transform res;
    float arr[4];
    _mm_store_ps(arr, res.rows_[0]);
    arr[3] = vec.x();
    res.rows_[0] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[1]);
    arr[3] = vec.y();
    res.rows_[1] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[2]);
    arr[3] = vec.z();
    res.rows_[2] = _mm_load_ps(arr);
    return res;
}
homogeneous_transform homogeneous_transform::scaling(
    const euclidean_coordinate& vec) {
    homogeneous_transform res;
    float arr[4];
    _mm_store_ps(arr, res.rows_[0]);
    arr[0] = vec.x();
    res.rows_[0] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[1]);
    arr[1] = vec.y();
    res.rows_[1] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[2]);
    arr[2] = vec.z();
    res.rows_[2] = _mm_load_ps(arr);
    return res;
}
homogeneous_transform homogeneous_transform::rotation(
    const euclidean_coordinate& axis, float angle) {
    euclidean_coordinate na = axis.normalize();
    float x = na.x(), y = na.y(), z = na.z();
    float c = std::cosf(angle), s = std::sinf(angle);
    float omc = 1.0f - c;
    homogeneous_transform res;
    float arr[4];

    arr[0] = c + x * x * omc;
    arr[1] = x * y * omc - z * s;
    arr[2] = x * z * omc + y * s;
    arr[3] = 0;
    res.rows_[0] = _mm_load_ps(arr);

    arr[0] = y * x * omc + z * s;
    arr[1] = c + y * y * omc;
    arr[2] = y * z * omc - x * s;
    arr[3] = 0;
    res.rows_[1] = _mm_load_ps(arr);

    arr[0] = z * x * omc - y * s;
    arr[1] = z * y * omc + x * s;
    arr[2] = c + z * z * omc;
    arr[3] = 0;
    res.rows_[2] = _mm_load_ps(arr);
    res.rows_[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return res;
}
bool homogeneous_transform::operator==(
    const homogeneous_transform& other) const {
    for (int i = 0; i < 4; ++i) {
        __m128 diff = _mm_sub_ps(rows_[i], other.rows_[i]);
        if (_mm_movemask_ps(_mm_cmpneq_ps(diff, _mm_setzero_ps())) != 0)
            return false;
    }
    return true;
}
homogeneous_transform homogeneous_transform::with_translation(
    const euclidean_coordinate& new_trans) const {
    homogeneous_transform res = *this;
    float arr[4];
    _mm_store_ps(arr, res.rows_[0]);
    arr[3] = new_trans.x();
    res.rows_[0] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[1]);
    arr[3] = new_trans.y();
    res.rows_[1] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[2]);
    arr[3] = new_trans.z();
    res.rows_[2] = _mm_load_ps(arr);
    return res;
}
homogeneous_transform homogeneous_transform::with_scale(
    const euclidean_coordinate& new_scale) const {
    homogeneous_transform res = *this;
    float arr[4];
    _mm_store_ps(arr, res.rows_[0]);
    arr[0] = new_scale.x();
    res.rows_[0] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[1]);
    arr[1] = new_scale.y();
    res.rows_[1] = _mm_load_ps(arr);
    _mm_store_ps(arr, res.rows_[2]);
    arr[2] = new_scale.z();
    res.rows_[2] = _mm_load_ps(arr);
    return res;
}
homogeneous_transform homogeneous_transform::then(
    const homogeneous_transform& after) const {
    return *this * after;
}
}  // namespace chrray