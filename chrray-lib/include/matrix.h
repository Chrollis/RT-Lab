#pragma once
#include <vector>

namespace chrray {
class homogeneous_coordinate;
class euclidean_coordinate {
public:
    euclidean_coordinate();
    euclidean_coordinate(std::initializer_list<double> values);
    euclidean_coordinate(double x, double y, double z);

    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    double& rx();
    double x() const;
    double& ry();
    double y() const;
    double& rz();
    double z() const;

    euclidean_coordinate operator+(const euclidean_coordinate& other) const;
    euclidean_coordinate operator-(const euclidean_coordinate& other) const;
    euclidean_coordinate operator*(double scalar) const;
    euclidean_coordinate operator/(double scalar) const;

    double dot(const euclidean_coordinate& other) const;
    euclidean_coordinate cross(const euclidean_coordinate& other) const;
    double length() const;
    euclidean_coordinate normalize() const;
    homogeneous_coordinate homogenize() const;

private:
    std::vector<double> container_;
};
class homogeneous_coordinate {
public:
    homogeneous_coordinate();
    homogeneous_coordinate(std::initializer_list<double> values);
    homogeneous_coordinate(double x, double y, double z, double w);

    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    double& rx();
    double x() const;
    double& ry();
    double y() const;
    double& rz();
    double z() const;
    double& rw();
    double w() const;

    homogeneous_coordinate operator+(const homogeneous_coordinate& other) const;
    homogeneous_coordinate operator-(const homogeneous_coordinate& other) const;
    homogeneous_coordinate operator*(double scalar) const;
    homogeneous_coordinate operator/(double scalar) const;

    double dot(const homogeneous_coordinate& other) const;
    double length() const;
    homogeneous_coordinate normalize() const;
    euclidean_coordinate dehomogenize() const;

private:
    std::vector<double> container_;
};
class quaternion {
public:
    quaternion();
    quaternion(std::initializer_list<double> values);
    quaternion(double x, double y, double z, double w);

    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    double& rx();
    double x() const;
    double& ry();
    double y() const;
    double& rz();
    double z() const;
    double& rw();
    double w() const;

    quaternion operator*(const quaternion& other) const;
    quaternion conjugate() const;
    double length() const;
    quaternion normalize() const;
    euclidean_coordinate rotate_vector(const euclidean_coordinate& vec) const;

private:
    std::vector<double> container_;

public:
    static quaternion from_axis_angle(
        const euclidean_coordinate& axis, double angle);
};
class homogeneous_transform {
public:
    homogeneous_transform();
    homogeneous_transform(
        std::initializer_list<std::initializer_list<double>> values);

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    homogeneous_transform operator*(const homogeneous_transform& other) const;
    homogeneous_coordinate operator*(const homogeneous_coordinate& vec) const;

private:
    std::vector<double> container_;

public:
    static homogeneous_transform translation(const euclidean_coordinate& vec);
    static homogeneous_transform scaling(const euclidean_coordinate& vec);
    static homogeneous_transform rotation(
        const euclidean_coordinate& axis, double angle);
};
}  // namespace chrray