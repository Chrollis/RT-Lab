#pragma once
#include <array>

namespace chrray {
class homogeneous_coordinate;
class euclidean_coordinate {
public:
    euclidean_coordinate();
    euclidean_coordinate(std::initializer_list<float> values);
    euclidean_coordinate(float x, float y, float z);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& rx();
    float x() const;
    float& ry();
    float y() const;
    float& rz();
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

private:
    std::array<float, 3> container_;
};
class homogeneous_coordinate {
public:
    homogeneous_coordinate();
    homogeneous_coordinate(std::initializer_list<float> values);
    homogeneous_coordinate(float x, float y, float z, float w);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& rx();
    float x() const;
    float& ry();
    float y() const;
    float& rz();
    float z() const;
    float& rw();
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

private:
    std::array<float, 4> container_;
};
class quaternion {
public:
    quaternion();
    quaternion(std::initializer_list<float> values);
    quaternion(float x, float y, float z, float w);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& rx();
    float x() const;
    float& ry();
    float y() const;
    float& rz();
    float z() const;
    float& rw();
    float w() const;

    quaternion operator*(const quaternion& other) const;
    quaternion conjugate() const;
    float length() const;
    quaternion normalize() const;
    euclidean_coordinate rotate_vector(const euclidean_coordinate& vec) const;
    bool operator==(const quaternion& other) const;

private:
    std::array<float, 4> container_;

public:
    static quaternion from_axis_angle(
        const euclidean_coordinate& axis, float angle);
};
class homogeneous_transform {
public:
    homogeneous_transform();
    homogeneous_transform(
        std::initializer_list<std::initializer_list<float>> values);

    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    homogeneous_transform operator*(const homogeneous_transform& other) const;
    homogeneous_coordinate operator*(const homogeneous_coordinate& vec) const;
    bool operator==(const homogeneous_transform& other) const;

private:
    std::array<float, 16> container_;

public:
    static homogeneous_transform translation(const euclidean_coordinate& vec);
    static homogeneous_transform scaling(const euclidean_coordinate& vec);
    static homogeneous_transform rotation(
        const euclidean_coordinate& axis, float angle);
};
}  // namespace chrray