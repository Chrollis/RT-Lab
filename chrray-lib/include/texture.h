#pragma once
#include <Windows.h>
#include <matrix.h>
#include <object.h>

namespace chrray {
struct color {
    double r, g, b, a;
    color();
    color(double r, double g, double b, double a);
    color(const euclidean_coordinate& rgb, double a);

    color operator+(const color& other) const;
    color operator*(const color& other) const;
    color operator*(double t) const;
    friend color operator*(double t, const color& c);

    color gamma_correct(double gamma) const;
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
    virtual double specular_strength(
        const hit_record& rec, const euclidean_coordinate& view_dir) const;
    virtual double shininess(const hit_record& rec) const;
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
    metal(const color& albedo, double fuzziness = 0.0);
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const override;
    virtual color specular_color(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual double specular_strength(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual double shininess(const hit_record& rec) const override;

private:
    color albedo_;
    double fuzziness_;
};

class dielectric : public material {
public:
    dielectric(
        double refraction_index,
        const color& attenuation = color(1, 1, 1, 1),
        double absorption_strength = 0.0);
    virtual bool scatter(
        const ray& in,
        const hit_record& rec,
        color& attenuation,
        ray& out) const override;
    virtual color specular_color(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual double specular_strength(
        const hit_record& rec,
        const euclidean_coordinate& view_dir) const override;
    virtual double shininess(const hit_record& rec) const override;

private:
    double refraction_index_;
    color attenuation_;
    double absorption_strength_;

    static double schlick(double cosine, double ri);
};
}  // namespace chrray
