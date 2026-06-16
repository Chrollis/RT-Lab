#pragma once
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <array>
#include <initializer_list>

namespace chrray {
class homogeneous_coordinate;
class euclidean_coordinate {
public:
    euclidean_coordinate();
    euclidean_coordinate(std::initializer_list<float> values);
    euclidean_coordinate(float x, float y, float z);

    float x() const;
    float y() const;
    float z() const;

    euclidean_coordinate operator+(const euclidean_coordinate& other) const;
    euclidean_coordinate operator-(const euclidean_coordinate& other) const;
    euclidean_coordinate operator*(float scalar) const;
    euclidean_coordinate operator/(float scalar) const;

    float dot(const euclidean_coordinate& other) const;
    euclidean_coordinate cross(const euclidean_coordinate& other) const;
    float length() const;
    euclidean_coordinate normalize() const;
    homogeneous_coordinate homogenize() const;

    euclidean_coordinate operator-() const;
    friend euclidean_coordinate operator*(
        float scalar, const euclidean_coordinate& vec);
    bool operator==(const euclidean_coordinate& other) const;

    euclidean_coordinate with_x(float new_x) const;
    euclidean_coordinate with_y(float new_y) const;
    euclidean_coordinate with_z(float new_z) const;

private:
    alignas(16) __m128 data_;
    explicit euclidean_coordinate(__m128 data);
};

class homogeneous_coordinate {
public:
    homogeneous_coordinate();
    homogeneous_coordinate(std::initializer_list<float> values);
    homogeneous_coordinate(float x, float y, float z, float w);

    float x() const;
    float y() const;
    float z() const;
    float w() const;

    homogeneous_coordinate operator+(const homogeneous_coordinate& other) const;
    homogeneous_coordinate operator-(const homogeneous_coordinate& other) const;
    homogeneous_coordinate operator*(float scalar) const;
    homogeneous_coordinate operator/(float scalar) const;

    float dot(const homogeneous_coordinate& other) const;
    float length() const;
    homogeneous_coordinate normalize() const;
    euclidean_coordinate dehomogenize() const;

    homogeneous_coordinate operator-() const;
    friend homogeneous_coordinate operator*(
        float scalar, const homogeneous_coordinate& vec);
    bool operator==(const homogeneous_coordinate& other) const;

    homogeneous_coordinate with_x(float new_x) const;
    homogeneous_coordinate with_y(float new_y) const;
    homogeneous_coordinate with_z(float new_z) const;
    homogeneous_coordinate with_w(float new_w) const;

private:
    alignas(16) __m128 data_;
    explicit homogeneous_coordinate(__m128 data);
};
class quaternion {
public:
    quaternion();
    quaternion(std::initializer_list<float> values);
    quaternion(float x, float y, float z, float w);

    float x() const;
    float y() const;
    float z() const;
    float w() const;

    quaternion operator*(const quaternion& other) const;
    quaternion conjugate() const;
    float length() const;
    quaternion normalize() const;
    euclidean_coordinate rotate_vector(const euclidean_coordinate& vec) const;
    bool operator==(const quaternion& other) const;

    quaternion with_x(float new_x) const;
    quaternion with_y(float new_y) const;
    quaternion with_z(float new_z) const;
    quaternion with_w(float new_w) const;

private:
    alignas(16) __m128 data_;
    explicit quaternion(__m128 data);

public:
    static quaternion from_axis_angle(
        const euclidean_coordinate& axis, float angle);
};
class homogeneous_transform {
public:
    homogeneous_transform();
    homogeneous_transform(
        std::initializer_list<std::initializer_list<float>> values);

    float operator()(size_t row, size_t col) const;

    homogeneous_transform operator*(const homogeneous_transform& other) const;
    homogeneous_coordinate operator*(const homogeneous_coordinate& vec) const;
    bool operator==(const homogeneous_transform& other) const;

    homogeneous_transform with_translation(
        const euclidean_coordinate& new_trans) const;
    homogeneous_transform with_scale(
        const euclidean_coordinate& new_scale) const;
    homogeneous_transform then(const homogeneous_transform& after) const;

private:
    alignas(16) std::array<__m128, 4> rows_;
    explicit homogeneous_transform(const std::array<__m128, 4>& rows);

public:
    static homogeneous_transform translation(const euclidean_coordinate& vec);
    static homogeneous_transform scaling(const euclidean_coordinate& vec);
    static homogeneous_transform rotation(
        const euclidean_coordinate& axis, float angle);
};
}  // namespace chrray