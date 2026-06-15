#include <texture.h>
#include <algorithm>

namespace chrray {
color::color() : r(0.0f), b(0.0f), g(0.0f), a(1.0f) {}
color::color(float r, float g, float b, float a) {
    this->r = std::clamp(r, 0.0f, 1.0f);
    this->g = std::clamp(g, 0.0f, 1.0f);
    this->b = std::clamp(b, 0.0f, 1.0f);
    this->a = std::clamp(a, 0.0f, 1.0f);
}
color::color(const euclidean_coordinate& rgb, float a) {
    this->r = std::clamp(rgb[0], 0.0f, 1.0f);
    this->g = std::clamp(rgb[1], 0.0f, 1.0f);
    this->b = std::clamp(rgb[2], 0.0f, 1.0f);
    this->a = std::clamp(a, 0.0f, 1.0f);
}

color color::operator+(const color& other) const {
    return color(r + other.r, g + other.g, b + other.b, a + other.a);
}
color color::operator*(const color& other) const {
    return color(r * other.r, g * other.g, b * other.b, a * other.a);
}
color color::operator*(float t) const {
    return color(r * t, g * t, b * t, a * t);
}
color operator*(float t, const color& c) {
    return color(c.r * t, c.g * t, c.b * t, c.a * t);
}

color color::gamma_correct(float gamma) const {
    float inv = 1.0f / gamma;
    return color(powf(r, inv), powf(g, inv), powf(b, inv), a);
}
color color::tonemap() const {
    float luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
    float mapped_luminance = luminance / (1.0f + luminance);
    float scale = mapped_luminance / (luminance + 1e-6f);
    return color(r * scale, g * scale, b * scale, a);
}
color blend(const color& c1, const color& c2) {
    return color(
        c1.r * c1.a + c2.r * c2.a * (1.0f - c1.a),
        c1.g * c1.a + c2.g * c2.a * (1.0f - c1.a),
        c1.b * c1.a + c2.b * c2.a * (1.0f - c1.a), c1.a + c2.a * (1.0f - c1.a));
}
COLORREF color::to_colorref(COLORREF bg) const {
    color blended = blend(
        *this, color(
                   GetRValue(bg) / 255.0f, GetGValue(bg) / 255.0f,
                   GetBValue(bg) / 255.0f, 1.0f));
    return RGB(
        static_cast<BYTE>(blended.r * 255.0f),
        static_cast<BYTE>(blended.g * 255.0f),
        static_cast<BYTE>(blended.b * 255.0f));
}

color color::black() {
    return color(0.0f, 0.0f, 0.0f, 1.0f);
}
color color::white() {
    return color(1.0f, 1.0f, 1.0f, 1.0f);
}
color color::gray() {
    return color(0.5f, 0.5f, 0.5f, 1.0f);
}
color color::dark_gray() {
    return color(0.25f, 0.25f, 0.25f, 1.0f);
}
color color::light_gray() {
    return color(0.75f, 0.75f, 0.75f, 1.0f);
}
color color::red() {
    return color(1.0f, 0.0f, 0.0f, 1.0f);
}
color color::dark_red() {
    return color(0.5f, 0.0f, 0.0f, 1.0f);
}
color color::light_red() {
    return color(1.0f, 0.5f, 0.5f, 1.0f);
}
color color::blue() {
    return color(0.0f, 0.0f, 1.0f, 1.0f);
}
color color::dark_blue() {
    return color(0.0f, 0.0f, 0.5f, 1.0f);
}
color color::light_blue() {
    return color(0.5f, 0.5f, 1.0f, 1.0f);
}
color color::green() {
    return color(0.0f, 1.0f, 0.0f, 1.0f);
}
color color::dark_green() {
    return color(0.0f, 0.5f, 0.0f, 1.0f);
}
color color::light_green() {
    return color(0.5f, 1.0f, 0.5f, 1.0f);
}
color color::yellow() {
    return color(1.0f, 1.0f, 0.0f, 1.0f);
}
color color::dark_yellow() {
    return color(0.5f, 0.5f, 0.0f, 1.0f);
}
color color::light_yellow() {
    return color(1.0f, 1.0f, 0.5f, 1.0f);
}
color color::magenta() {
    return color(1.0f, 0.0f, 1.0f, 1.0f);
}
color color::dark_magenta() {
    return color(0.5f, 0.0f, 0.5f, 1.0f);
}
color color::light_magenta() {
    return color(1.0f, 0.5f, 1.0f, 1.0f);
}
color color::cyan() {
    return color(0.0f, 1.0f, 1.0f, 1.0f);
}
color color::dark_cyan() {
    return color(0.0f, 0.5f, 0.5f, 1.0f);
}
color color::light_cyan() {
    return color(0.5f, 1.0f, 1.0f, 1.0f);
}

color material::diffuse_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return color::black();
}
color material::specular_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return color::black();
}
float material::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return 0.0f;
}
float material::shininess(const hit_record& rec) const {
    return 1.0f;
}
color material::emitted(const hit_record& rec) const {
    return color::black();
}

lambertian::lambertian(const color& albedo) : albedo_(albedo) {}
bool lambertian::scatter(
    const ray& in, const hit_record& rec, color& attenuation, ray& out) const {
    euclidean_coordinate scatter_dir = rec.n + random_on_hemisphere(rec.n);
    if (scatter_dir.length() < eps) {
        scatter_dir = rec.n;
    }
    out = ray(rec.p, scatter_dir.normalize());
    attenuation = albedo_;
    return true;
}
color lambertian::diffuse_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return albedo_;
}

metal::metal(const color& albedo, float fuzziness)
    : albedo_(albedo), fuzziness_(std::clamp(fuzziness, 0.0f, 1.0f)) {}
bool metal::scatter(
    const ray& in, const hit_record& rec, color& attenuation, ray& out) const {
    euclidean_coordinate reflected = reflect(in.direction().normalize(), rec.n);
    if (fuzziness_ > 0.0f) {
        reflected = reflected + fuzziness_ * random_in_unit_sphere();
    }
    if (reflected.dot(rec.n) <= 0.0f) {
        return false;
    }
    out = ray(rec.p, reflected.normalize());
    attenuation = albedo_;
    return true;
}
color metal::diffuse_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return albedo_;
}
color metal::specular_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return albedo_;
}
float metal::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return 1.0f - fuzziness_ * 0.5f;
}
float metal::shininess(const hit_record& rec) const {
    float shine = 1.0f / (fuzziness_ + 1e-4f);
    if (shine > 1000.0f) shine = 1000.0f;
    return shine;
}

dielectric::dielectric(
    float refraction_index, const color& attenuation, float absorption_strength)
    : refraction_index_(refraction_index),
      attenuation_(attenuation),
      absorption_strength_(absorption_strength) {}
bool dielectric::scatter(
    const ray& in, const hit_record& rec, color& attenuation, ray& out) const {
    euclidean_coordinate outward_normal;
    float ni_over_nt;
    float cosine;
    float reflect_prob;

    if (in.direction().dot(rec.n) < 0.0f) {
        outward_normal = rec.n;
        ni_over_nt = 1.0f / refraction_index_;
        cosine = -in.direction().dot(rec.n);
    } else {
        outward_normal = -rec.n;
        ni_over_nt = refraction_index_;
        cosine = refraction_index_ * in.direction().dot(rec.n);
    }

    euclidean_coordinate refracted;
    if (refract(in.direction(), outward_normal, ni_over_nt, refracted)) {
        reflect_prob = schlick(cosine, refraction_index_);
    } else {
        reflect_prob = 1.0f;
    }

    if (random_float() < reflect_prob) {
        euclidean_coordinate reflected =
            reflect(in.direction().normalize(), rec.n);
        out = ray(rec.p, reflected);
        attenuation = color(1, 1, 1, 1);
    } else {
        out = ray(rec.p, refracted);
        float t_inside = (rec.p - out.origin()).length();
        float absorption = std::exp(-absorption_strength_ * t_inside);
        attenuation = color(
            attenuation_.r * absorption, attenuation_.g * absorption,
            attenuation_.b * absorption, 1);
    }
    return true;
}
color dielectric::specular_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return color::white();
}

float dielectric::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    float cos_theta = std::abs(rec.n.dot(view_dir));
    float eta = 1.0f / refraction_index_;
    float reflectance = schlick(cos_theta, eta);
    return reflectance;
}

float dielectric::shininess(const hit_record& rec) const {
    return 200.0f;
}
float dielectric::schlick(float cosine, float ri) {
    float r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::powf(1 - cosine, 5);
}

}  // namespace chrray