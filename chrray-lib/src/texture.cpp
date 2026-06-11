#include <texture.h>
#include <algorithm>

namespace chrray {
color::color() : r(0.0), b(0.0), g(0.0), a(1.0) {}
color::color(double r, double g, double b, double a) {
    this->r = std::clamp(r, 0.0, 1.0);
    this->g = std::clamp(g, 0.0, 1.0);
    this->b = std::clamp(b, 0.0, 1.0);
    this->a = std::clamp(a, 0.0, 1.0);
}
color::color(const euclidean_coordinate& rgb, double a) {
    this->r = std::clamp(rgb[0], 0.0, 1.0);
    this->g = std::clamp(rgb[1], 0.0, 1.0);
    this->b = std::clamp(rgb[2], 0.0, 1.0);
    this->a = std::clamp(a, 0.0, 1.0);
}

color color::operator+(const color& other) const {
    return color(r + other.r, g + other.g, b + other.b, a + other.a);
}
color color::operator*(const color& other) const {
    return color(r * other.r, g * other.g, b * other.b, a * other.a);
}
color color::operator*(double t) const {
    return color(r * t, g * t, b * t, a * t);
}
color operator*(double t, const color& c) {
    return color(c.r * t, c.g * t, c.b * t, c.a * t);
}

color color::gamma_correct(double gamma) const {
    double inv = 1.0 / gamma;
    return color(pow(r, inv), pow(g, inv), powf(b, inv), a);
}
color blend(const color& c1, const color& c2) {
    return color(
        c1.r * c1.a + c2.r * c2.a * (1.0 - c1.a),
        c1.g * c1.a + c2.g * c2.a * (1.0 - c1.a),
        c1.b * c1.a + c2.b * c2.a * (1.0 - c1.a), c1.a + c2.a * (1.0 - c1.a));
}
COLORREF color::to_colorref(COLORREF bg) const {
    color blended = blend(
        *this, color(
                   GetRValue(bg) / 255.0, GetGValue(bg) / 255.0,
                   GetBValue(bg) / 255.0, 1.0));
    return RGB(
        static_cast<BYTE>(blended.r * 255.0),
        static_cast<BYTE>(blended.g * 255.0),
        static_cast<BYTE>(blended.b * 255.0));
}

color color::black() {
    return color(0.0, 0.0, 0.0, 1.0);
}
color color::white() {
    return color(1.0, 1.0, 1.0, 1.0);
}
color color::gray() {
    return color(0.5, 0.5, 0.5, 1.0);
}
color color::dark_gray() {
    return color(0.25, 0.25, 0.25, 1.0);
}
color color::light_gray() {
    return color(0.75, 0.75, 0.75, 1.0);
}
color color::red() {
    return color(1.0, 0.0, 0.0, 1.0);
}
color color::dark_red() {
    return color(0.5, 0.0, 0.0, 1.0);
}
color color::light_red() {
    return color(1.0, 0.5, 0.5, 1.0);
}
color color::blue() {
    return color(0.0, 0.0, 1.0, 1.0);
}
color color::dark_blue() {
    return color(0.0, 0.0, 0.5, 1.0);
}
color color::light_blue() {
    return color(0.5, 0.5, 1.0, 1.0);
}
color color::green() {
    return color(0.0, 1.0, 0.0, 1.0);
}
color color::dark_green() {
    return color(0.0, 0.5, 0.0, 1.0);
}
color color::light_green() {
    return color(0.5, 1.0, 0.5, 1.0);
}
color color::yellow() {
    return color(1.0, 1.0, 0.0, 1.0);
}
color color::dark_yellow() {
    return color(0.5, 0.5, 0.0, 1.0);
}
color color::light_yellow() {
    return color(1.0, 1.0, 0.5, 1.0);
}
color color::magenta() {
    return color(1.0, 0.0, 1.0, 1.0);
}
color color::dark_magenta() {
    return color(0.5, 0.0, 0.5, 1.0);
}
color color::light_magenta() {
    return color(1.0, 0.5, 1.0, 1.0);
}
color color::cyan() {
    return color(0.0, 1.0, 1.0, 1.0);
}
color color::dark_cyan() {
    return color(0.0, 0.5, 0.5, 1.0);
}
color color::light_cyan() {
    return color(0.5, 1.0, 1.0, 1.0);
}

color material::diffuse_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return color::black();
}
color material::specular_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return color::black();
}
double material::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return 0.0;
}
double material::shininess(const hit_record& rec) const {
    return 1.0;
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

metal::metal(const color& albedo, double fuzziness)
    : albedo_(albedo), fuzziness_(std::clamp(fuzziness, 0.0, 1.0)) {}
bool metal::scatter(
    const ray& in, const hit_record& rec, color& attenuation, ray& out) const {
    euclidean_coordinate reflected = reflect(in.direction().normalize(), rec.n);
    if (fuzziness_ > 0.0) {
        reflected = reflected + fuzziness_ * random_in_unit_sphere();
    }
    if (reflected.dot(rec.n) <= 0.0) {
        return false;
    }
    out = ray(rec.p, reflected.normalize());
    attenuation = albedo_;
    return true;
}
color metal::specular_color(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return albedo_;
}
double metal::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    return 1.0 - fuzziness_ * 0.5;
}
double metal::shininess(const hit_record& rec) const {
    double shine = 1.0 / (fuzziness_ + 1e-4);
    if (shine > 1000.0) shine = 1000.0;
    return shine;
}

dielectric::dielectric(
    double refraction_index,
    const color& attenuation,
    double absorption_strength)
    : refraction_index_(refraction_index),
      attenuation_(attenuation),
      absorption_strength_(absorption_strength) {}
bool dielectric::scatter(
    const ray& in, const hit_record& rec, color& attenuation, ray& out) const {
    euclidean_coordinate outward_normal;
    double ni_over_nt;
    double cosine;
    double reflect_prob;

    if (in.direction().dot(rec.n) < 0.0) {
        outward_normal = rec.n;
        ni_over_nt = 1.0 / refraction_index_;
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
        reflect_prob = 1.0;
    }

    if (random_double() < reflect_prob) {
        euclidean_coordinate reflected =
            reflect(in.direction().normalize(), rec.n);
        out = ray(rec.p, reflected);
        attenuation = color(1, 1, 1, 1);
    } else {
        out = ray(rec.p, refracted);
        double t_inside = (rec.p - out.origin()).length();
        double absorption = std::exp(-absorption_strength_ * t_inside);
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

double dielectric::specular_strength(
    const hit_record& rec, const euclidean_coordinate& view_dir) const {
    double cos_theta = std::abs(rec.n.dot(view_dir));
    double eta = 1.0 / refraction_index_;
    double reflectance = schlick(cos_theta, eta);
    return reflectance;
}

double dielectric::shininess(const hit_record& rec) const {
    return 200.0;
}
double dielectric::schlick(double cosine, double ri) {
    double r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

}  // namespace chrray