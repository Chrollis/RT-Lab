#include <matrix.h>
#include <cmath>
#include <stdexcept>

namespace chrray {
euclidean_coordinate::euclidean_coordinate() : container_{0, 0, 0} {}
euclidean_coordinate::euclidean_coordinate(
    std::initializer_list<float> values) {
    if (container_.size() != 3) {
        throw std::invalid_argument(
            "Euclidean coordinate must have 3 components");
    }
    std::copy(values.begin(), values.end(), container_.begin());
}
euclidean_coordinate::euclidean_coordinate(float x, float y, float z)
    : container_{x, y, z} {}
float& euclidean_coordinate::operator[](size_t index) {
    return container_[index];
}
const float& euclidean_coordinate::operator[](size_t index) const {
    return container_[index];
}
float& euclidean_coordinate::rx() {
    return container_[0];
}
float euclidean_coordinate::x() const {
    return container_[0];
}
float& euclidean_coordinate::ry() {
    return container_[1];
}
float euclidean_coordinate::y() const {
    return container_[1];
}
float& euclidean_coordinate::rz() {
    return container_[2];
}
float euclidean_coordinate::z() const {
    return container_[2];
}
euclidean_coordinate euclidean_coordinate::operator+(
    const euclidean_coordinate& other) const {
    return euclidean_coordinate(
        container_[0] + other.container_[0],
        container_[1] + other.container_[1],
        container_[2] + other.container_[2]);
}
euclidean_coordinate euclidean_coordinate::operator-(
    const euclidean_coordinate& other) const {
    return euclidean_coordinate(
        container_[0] - other.container_[0],
        container_[1] - other.container_[1],
        container_[2] - other.container_[2]);
}
euclidean_coordinate euclidean_coordinate::operator*(float scalar) const {
    return euclidean_coordinate(
        container_[0] * scalar, container_[1] * scalar, container_[2] * scalar);
}
euclidean_coordinate euclidean_coordinate::operator/(float scalar) const {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero");
    }
    return euclidean_coordinate(
        container_[0] / scalar, container_[1] / scalar, container_[2] / scalar);
}
float euclidean_coordinate::dot(const euclidean_coordinate& other) const {
    return container_[0] * other.container_[0] +
           container_[1] * other.container_[1] +
           container_[2] * other.container_[2];
}
euclidean_coordinate euclidean_coordinate::cross(
    const euclidean_coordinate& other) const {
    return euclidean_coordinate(
        container_[1] * other.container_[2] -
            container_[2] * other.container_[1],
        container_[2] * other.container_[0] -
            container_[0] * other.container_[2],
        container_[0] * other.container_[1] -
            container_[1] * other.container_[0]);
}
float euclidean_coordinate::length() const {
    return std::sqrtf(
        container_[0] * container_[0] + container_[1] * container_[1] +
        container_[2] * container_[2]);
}
euclidean_coordinate euclidean_coordinate::normalize() const {
    float len = length();
    if (len == 0) {
        throw std::invalid_argument("Cannot normalize a zero-length vector");
    }
    return *this / len;
}
homogeneous_coordinate euclidean_coordinate::homogenize() const {
    return homogeneous_coordinate(
        container_[0], container_[1], container_[2], 1);
}
euclidean_coordinate euclidean_coordinate::operator-() const {
    return euclidean_coordinate(-container_[0], -container_[1], -container_[2]);
}
euclidean_coordinate operator*(float scalar, const euclidean_coordinate& vec) {
    return euclidean_coordinate(
        vec.container_[0] * scalar, vec.container_[1] * scalar,
        vec.container_[2] * scalar);
}
bool euclidean_coordinate::operator==(const euclidean_coordinate& other) const {
    return container_[0] == other.container_[0] &&
           container_[1] == other.container_[1] &&
           container_[2] == other.container_[2];
}

homogeneous_coordinate::homogeneous_coordinate() : container_{0, 0, 0, 0} {}
homogeneous_coordinate::homogeneous_coordinate(
    std::initializer_list<float> values) {
    if (container_.size() != 4) {
        throw std::invalid_argument(
            "Homogeneous coordinate must have 4 components");
    }
    std::copy(values.begin(), values.end(), container_.begin());
}
homogeneous_coordinate::homogeneous_coordinate(
    float x, float y, float z, float w)
    : container_{x, y, z, w} {}
float& homogeneous_coordinate::operator[](size_t index) {
    return container_[index];
}
const float& homogeneous_coordinate::operator[](size_t index) const {
    return container_[index];
}
float& homogeneous_coordinate::rx() {
    return container_[0];
}
float homogeneous_coordinate::x() const {
    return container_[0];
}
float& homogeneous_coordinate::ry() {
    return container_[1];
}
float homogeneous_coordinate::y() const {
    return container_[1];
}
float& homogeneous_coordinate::rz() {
    return container_[2];
}
float homogeneous_coordinate::z() const {
    return container_[2];
}
float& homogeneous_coordinate::rw() {
    return container_[3];
}
float homogeneous_coordinate::w() const {
    return container_[3];
}
homogeneous_coordinate homogeneous_coordinate::operator+(
    const homogeneous_coordinate& other) const {
    return homogeneous_coordinate(
        container_[0] + other.container_[0],
        container_[1] + other.container_[1],
        container_[2] + other.container_[2],
        container_[3] + other.container_[3]);
}
homogeneous_coordinate homogeneous_coordinate::operator-(
    const homogeneous_coordinate& other) const {
    return homogeneous_coordinate(
        container_[0] - other.container_[0],
        container_[1] - other.container_[1],
        container_[2] - other.container_[2],
        container_[3] - other.container_[3]);
}
homogeneous_coordinate homogeneous_coordinate::operator*(float scalar) const {
    return homogeneous_coordinate(
        container_[0] * scalar, container_[1] * scalar, container_[2] * scalar,
        container_[3] * scalar);
}
homogeneous_coordinate homogeneous_coordinate::operator/(float scalar) const {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero");
    }
    return homogeneous_coordinate(
        container_[0] / scalar, container_[1] / scalar, container_[2] / scalar,
        container_[3] / scalar);
}
float homogeneous_coordinate::dot(const homogeneous_coordinate& other) const {
    return container_[0] * other.container_[0] +
           container_[1] * other.container_[1] +
           container_[2] * other.container_[2] +
           container_[3] * other.container_[3];
}
float homogeneous_coordinate::length() const {
    return std::sqrtf(
        container_[0] * container_[0] + container_[1] * container_[1] +
        container_[2] * container_[2] + container_[3] * container_[3]);
}
homogeneous_coordinate homogeneous_coordinate::normalize() const {
    float len = length();
    if (len == 0) {
        throw std::invalid_argument("Cannot normalize a zero-length vector");
    }
    return *this / len;
}
euclidean_coordinate homogeneous_coordinate::dehomogenize() const {
    if (container_[3] == 0) {
        throw std::invalid_argument(
            "Cannot dehomogenize a point with w equal to zero");
    }
    return euclidean_coordinate(
        container_[0] / container_[3], container_[1] / container_[3],
        container_[2] / container_[3]);
}
homogeneous_coordinate homogeneous_coordinate::operator-() const {
    return homogeneous_coordinate(
        -container_[0], -container_[1], -container_[2], -container_[3]);
}
homogeneous_coordinate operator*(
    float scalar, const homogeneous_coordinate& vec) {
    return homogeneous_coordinate(
        vec.container_[0] * scalar, vec.container_[1] * scalar,
        vec.container_[2] * scalar, vec.container_[3] * scalar);
}
bool homogeneous_coordinate::operator==(
    const homogeneous_coordinate& other) const {
    return container_[0] == other.container_[0] &&
           container_[1] == other.container_[1] &&
           container_[2] == other.container_[2] &&
           container_[3] == other.container_[3];
}

quaternion::quaternion() : container_{0, 0, 0, 0} {}
quaternion::quaternion(std::initializer_list<float> values) {
    if (container_.size() != 4) {
        throw std::invalid_argument("Quaternion must have 4 components");
    }
    std::copy(values.begin(), values.end(), container_.begin());
}
quaternion::quaternion(float x, float y, float z, float w)
    : container_{x, y, z, w} {}
float& quaternion::operator[](size_t index) {
    return container_[index];
}
const float& quaternion::operator[](size_t index) const {
    return container_[index];
}
float& quaternion::rx() {
    return container_[0];
}
float quaternion::x() const {
    return container_[0];
}
float& quaternion::ry() {
    return container_[1];
}
float quaternion::y() const {
    return container_[1];
}
float& quaternion::rz() {
    return container_[2];
}
float quaternion::z() const {
    return container_[2];
}
float& quaternion::rw() {
    return container_[3];
}
float quaternion::w() const {
    return container_[3];
}
quaternion quaternion::operator*(const quaternion& other) const {
    return quaternion(
        container_[3] * other.container_[0] +
            container_[0] * other.container_[3] +
            container_[1] * other.container_[2] -
            container_[2] * other.container_[1],
        container_[3] * other.container_[1] -
            container_[0] * other.container_[2] +
            container_[1] * other.container_[3] +
            container_[2] * other.container_[0],
        container_[3] * other.container_[2] +
            container_[0] * other.container_[1] -
            container_[1] * other.container_[0] +
            container_[2] * other.container_[3],
        container_[3] * other.container_[3] -
            container_[0] * other.container_[0] -
            container_[1] * other.container_[1] -
            container_[2] * other.container_[2]);
}
quaternion quaternion::conjugate() const {
    return quaternion(
        -container_[0], -container_[1], -container_[2], container_[3]);
}
float quaternion::length() const {
    return std::sqrtf(
        container_[0] * container_[0] + container_[1] * container_[1] +
        container_[2] * container_[2] + container_[3] * container_[3]);
}
quaternion quaternion::normalize() const {
    float len = length();
    if (len == 0) {
        throw std::invalid_argument(
            "Cannot normalize a zero-length quaternion");
    }
    return quaternion(
        container_[0] / len, container_[1] / len, container_[2] / len,
        container_[3] / len);
}
euclidean_coordinate quaternion::rotate_vector(
    const euclidean_coordinate& vec) const {
    quaternion normalized = normalize();
    quaternion vector_quat(vec.x(), vec.y(), vec.z(), 0);
    quaternion rotated = normalized * vector_quat * normalized.conjugate();
    return euclidean_coordinate(rotated.x(), rotated.y(), rotated.z());
}
quaternion quaternion::from_axis_angle(
    const euclidean_coordinate& axis, float angle) {
    euclidean_coordinate normalized_axis = axis.normalize();
    float half_angle = angle * 0.5f;
    float sin_half = std::sinf(half_angle);
    return quaternion(
        normalized_axis.x() * sin_half, normalized_axis.y() * sin_half,
        normalized_axis.z() * sin_half, std::cosf(half_angle));
}
bool quaternion::operator==(const quaternion& other) const {
    return container_[0] == other.container_[0] &&
           container_[1] == other.container_[1] &&
           container_[2] == other.container_[2] &&
           container_[3] == other.container_[3];
}

homogeneous_transform::homogeneous_transform()
    : container_{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} {}
homogeneous_transform::homogeneous_transform(
    std::initializer_list<std::initializer_list<float>> values) {
    if (values.size() != 4) {
        throw std::invalid_argument("Homogeneous transform must be 4x4");
    }
    size_t row = 0;
    for (const auto& row_values : values) {
        if (row_values.size() != 4) {
            throw std::invalid_argument("Homogeneous transform must be 4x4");
        }
        size_t col = 0;
        for (float value : row_values) {
            container_[row * 4 + col] = value;
            ++col;
        }
        ++row;
    }
}
float& homogeneous_transform::operator()(size_t row, size_t col) {
    return container_[row * 4 + col];
}
const float& homogeneous_transform::operator()(size_t row, size_t col) const {
    return container_[row * 4 + col];
}
float& homogeneous_transform::operator[](size_t index) {
    return container_[index];
}
const float& homogeneous_transform::operator[](size_t index) const {
    return container_[index];
}
homogeneous_transform homogeneous_transform::operator*(
    const homogeneous_transform& other) const {
    homogeneous_transform result;
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            float value = 0;
            for (size_t k = 0; k < 4; ++k) {
                value += (*this)(row, k) * other(k, col);
            }
            result(row, col) = value;
        }
    }
    return result;
}
homogeneous_coordinate homogeneous_transform::operator*(
    const homogeneous_coordinate& vec) const {
    return homogeneous_coordinate(
        (*this)(0, 0) * vec.x() + (*this)(0, 1) * vec.y() +
            (*this)(0, 2) * vec.z() + (*this)(0, 3) * vec.w(),
        (*this)(1, 0) * vec.x() + (*this)(1, 1) * vec.y() +
            (*this)(1, 2) * vec.z() + (*this)(1, 3) * vec.w(),
        (*this)(2, 0) * vec.x() + (*this)(2, 1) * vec.y() +
            (*this)(2, 2) * vec.z() + (*this)(2, 3) * vec.w(),
        (*this)(3, 0) * vec.x() + (*this)(3, 1) * vec.y() +
            (*this)(3, 2) * vec.z() + (*this)(3, 3) * vec.w());
}
homogeneous_transform homogeneous_transform::translation(
    const euclidean_coordinate& vec) {
    homogeneous_transform result;
    result(0, 3) = vec.x();
    result(1, 3) = vec.y();
    result(2, 3) = vec.z();
    return result;
}
homogeneous_transform homogeneous_transform::scaling(
    const euclidean_coordinate& vec) {
    homogeneous_transform result;
    result(0, 0) = vec.x();
    result(1, 1) = vec.y();
    result(2, 2) = vec.z();
    return result;
}
homogeneous_transform homogeneous_transform::rotation(
    const euclidean_coordinate& axis, float angle) {
    euclidean_coordinate normalized_axis = axis.normalize();
    float x = normalized_axis.x();
    float y = normalized_axis.y();
    float z = normalized_axis.z();
    float c = std::cosf(angle);
    float s = std::sinf(angle);
    float one_minus_c = 1 - c;

    homogeneous_transform result;
    result(0, 0) = c + x * x * one_minus_c;
    result(0, 1) = x * y * one_minus_c - z * s;
    result(0, 2) = x * z * one_minus_c + y * s;
    result(0, 3) = 0;

    result(1, 0) = y * x * one_minus_c + z * s;
    result(1, 1) = c + y * y * one_minus_c;
    result(1, 2) = y * z * one_minus_c - x * s;
    result(1, 3) = 0;

    result(2, 0) = z * x * one_minus_c - y * s;
    result(2, 1) = z * y * one_minus_c + x * s;
    result(2, 2) = c + z * z * one_minus_c;
    result(2, 3) = 0;

    result(3, 0) = 0;
    result(3, 1) = 0;
    result(3, 2) = 0;
    result(3, 3) = 1;
    return result;
}
bool homogeneous_transform::operator==(
    const homogeneous_transform& other) const {
    for (size_t i = 0; i < 16; ++i) {
        if (container_[i] != other.container_[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace chrray