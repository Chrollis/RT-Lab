#pragma once
#include <Windows.h>
#include <matrix.h>
#include <object.h>

namespace chrray {
struct color {
    float r, g, b, a;
    color();
    color(float r, float g, float b, float a);
    color(const euclidean_coordinate& rgb, float a);

    color operator+(const color& other) const;
    color operator*(const color& other) const;
    color operator*(float t) const;
    friend color operator*(float t, const color& c);

    color gamma_correct(float gamma) const;
    friend color blend(const color& c1, const color& c2);
    COLORREF to_colorref(COLORREF bg) const;

    static color black();
    static color white();
    static color gray();
    static color dark_gray();
    static color light_gray();

    static color red();
    static color dark_red();
    static color light_red();
    static color blue();
    static color dark_blue();
    static color light_blue();
    static color green();
    static color dark_green();
    static color light_green();

    static color yellow();
    static color dark_yellow();
    static color light_yellow();
    static color magenta();
    static color dark_magenta();
    static color light_magenta();
    static color cyan();
    static color dark_cyan();
    static color light_cyan();
};

class material {
public:
    virtual ~material() = default;
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const = 0;
    virtual color diffuse_color(
        const hit_record& rec, const euclidean_coordinate& view_dir) const;
    virtual color specular_color(
        const hit_record& rec, const euclidean_coordinate& view_dir) const;
    virtual float specular_strength(
        const hit_record& rec, const euclidean_coordinate& view_dir) const;
    virtual float shininess(const hit_record& rec) const;
    virtual color emitted(const hit_record& rec) const;
};

class lambertian : public material {
public:
    lambertian(const color& albedo);
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const override;
    virtual color diffuse_color(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;

private:
    color albedo_;
};

class metal : public material {
public:
    metal(const color& albedo, float fuzziness = 0.0f);
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const override;
    virtual color specular_color(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual float specular_strength(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual float shininess(const hit_record& rec) const override;

private:
    color albedo_;
    float fuzziness_;
};

class dielectric : public material {
public:
    dielectric(
        float refraction_index,
        const color& attenuation = color(1, 1, 1, 1),
        float absorption_strength = 0.0f);
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const override;
    virtual color specular_color(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual float specular_strength(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual float shininess(const hit_record& rec) const override;

private:
    float refraction_index_;
    color attenuation_;
    float absorption_strength_;

    static float schlick(float cosine, float ri);
};
}  // namespace chrray
