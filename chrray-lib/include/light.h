#pragma once
#include <matrix.h>
#include <object.h>
#include <texture.h>
#include <vector>

namespace chrray {

class light {
public:
    virtual ~light() = default;
    virtual euclidean_coordinate direction(
        const euclidean_coordinate& hit_point) const = 0;
    virtual double distance(const euclidean_coordinate& hit_point) const = 0;
    virtual color intensity(const euclidean_coordinate& hit_point) const = 0;
    virtual ray shadow_ray(const euclidean_coordinate& hit_point) const = 0;
};

class point_light : public light {
public:
    point_light(
        const euclidean_coordinate& position,
        const color& intensity,
        double constant_atten = 1.0,
        double linear_atten = 0.0,
        double quad_atten = 0.0);

    euclidean_coordinate direction(
        const euclidean_coordinate& hit_point) const override;
    double distance(const euclidean_coordinate& hit_point) const override;
    color intensity(const euclidean_coordinate& hit_point) const override;
    ray shadow_ray(const euclidean_coordinate& hit_point) const override;

private:
    euclidean_coordinate position_;
    color intensity_;
    double const_atten_;
    double lin_atten_;
    double quad_atten_;
};

class directional_light : public light {
public:
    directional_light(
        const euclidean_coordinate& direction, const color& intensity);

    euclidean_coordinate direction(
        const euclidean_coordinate& hit_point) const override;
    double distance(const euclidean_coordinate& hit_point) const override;
    color intensity(const euclidean_coordinate& hit_point) const override;
    ray shadow_ray(const euclidean_coordinate& hit_point) const override;

private:
    euclidean_coordinate direction_;
    color intensity_;
};

}  // namespace chrray