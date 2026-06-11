#include <bvh_node.h>
#include <camera.h>
#include <graphics.h>
#include <light.h>
#include <object.h>
#include <omp.h>
#include <texture.h>
#include <utils.h>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#undef max
#undef min

using namespace chrray;

const int WIDTH = 1600;
const int HEIGHT = 900;
const int MIN_DEPTH = 8;
const float SURVIVAL = 0.8f;
const int MAX_DEPTH = 16;
const int NUM_OBJECTS = 200;
const color BACKGROUND_COLOR(0.1f, 0.15f, 0.25f, 1.0f);

std::vector<std::shared_ptr<hittable>> world;
std::vector<std::shared_ptr<light>> lights;
std::shared_ptr<bvh_node> bvh_root;
bool use_bvh = false;

camera cam(
    euclidean_coordinate(0, 10, 25),
    euclidean_coordinate(0, 5, 0),
    euclidean_coordinate(0, 1, 0),
    60.0f,
    float(WIDTH) / HEIGHT,
    0.1f,
    100.0f);

bool is_shadowed(
    const euclidean_coordinate& hit_point,
    const light* lit,
    float light_distance) {
    ray shadow_ray = lit->shadow_ray(hit_point);
    float t_max = (light_distance > 0) ? light_distance : inf;
    hit_record rec;

    if (use_bvh) {
        return bvh_root->any_hit(shadow_ray, eps, t_max);
    }
    for (const auto& obj : world) {
        if (obj->intersect(shadow_ray, eps, t_max, rec)) return true;
    }
    return false;
}

color ray_color(const ray& r, int depth) {
    if (depth >= MAX_DEPTH) return color::black();

    hit_record rec;
    float closest_t = inf;
    std::shared_ptr<hittable> hit_obj = nullptr;

    if (use_bvh) {
        if (bvh_root->intersect(r, eps, inf, rec)) {
            hit_obj = bvh_root;
            closest_t = rec.t;
        }
    } else {
        for (const auto& obj : world) {
            if (obj->intersect(r, eps, inf, rec)) {
                if (rec.t < closest_t) {
                    closest_t = rec.t;
                    hit_obj = obj;
                }
            }
        }
    }

    if (!hit_obj) {
        float t = 0.5f * (r.direction().y() + 1.0f);
        return BACKGROUND_COLOR * (1.0f - t) +
               color(0.5f, 0.7f, 1.0f, 1.0f) * t;
    }

    if (!use_bvh) hit_obj->intersect(r, eps, inf, rec);

    euclidean_coordinate view_dir = -r.direction().normalize();

    color direct(0, 0, 0, 1);
    for (const auto& lit : lights) {
        euclidean_coordinate light_dir = lit->direction(rec.p);
        float light_dist = lit->distance(rec.p);
        if (is_shadowed(rec.p, lit.get(), light_dist)) continue;

        color light_intensity = lit->intensity(rec.p);
        float ndotl = std::max(0.0f, rec.n.dot(light_dir));
        color diff =
            rec.mat->diffuse_color(rec, view_dir) * light_intensity * ndotl;

        euclidean_coordinate half_dir = (light_dir + view_dir).normalize();
        float ndoth = std::max(0.0f, rec.n.dot(half_dir));
        float shininess = rec.mat->shininess(rec);
        float spec_intensity = rec.mat->specular_strength(rec, view_dir) *
                               std::powf(ndoth, shininess);
        color spec = rec.mat->specular_color(rec, view_dir) * light_intensity *
                     spec_intensity;

        direct = direct + diff + spec;
    }

    color ambient = rec.mat->diffuse_color(rec, view_dir) * 0.1f;

    color scattered_color(0, 0, 0, 1);
    if (depth < MAX_DEPTH) {
        if (depth > MIN_DEPTH && random_float() > SURVIVAL) {
            return ambient + direct;
        }
        ray scattered;
        color attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered)) {
            scattered_color = ray_color(scattered, depth + 1) * attenuation;
        }
    }

    color emitted = rec.mat->emitted(rec);

    color final = ambient + direct + scattered_color + emitted;
    return final;
}

void render() {
    std::vector<color> framebuffer(WIDTH * HEIGHT);
#pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < WIDTH * HEIGHT; ++idx) {
        int x = idx % WIDTH;
        int y = idx / WIDTH;
        float u = float(x) / (WIDTH - 1);
        float v = float(y) / (HEIGHT - 1);
        ray r = cam.get_ray(u, v);
        color col = ray_color(r, 0);
        col = col.gamma_correct(2.2f);
        framebuffer[idx] = col;
    }

    BeginBatchDraw();
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const color& col = framebuffer[y * WIDTH + x];
            putpixel(
                x, y,
                RGB(int(col.r * 255), int(col.g * 255), int(col.b * 255)));
        }
        if (y % 20 == 0) FlushBatchDraw();
    }
    EndBatchDraw();
}

void build_scene() {
    world.clear();
    lights.clear();

    auto na_metal =
        std::make_shared<metal>(color(1.0f, 1.0f, 0.9f, 1.0f), 0.05f);
    auto na_lambertian =
        std::make_shared<lambertian>(color(1.0f, 1.0f, 0.9f, 1.0f));
    auto na_dielectric = std::make_shared<dielectric>(
        1.5f, color(1.0f, 1.0f, 0.9f, 1.0f), 0.01f);
    auto al_lambertian =
        std::make_shared<lambertian>(color(1.0f, 1.0f, 1.0f, 1.0f));
    auto f_lambertian =
        std::make_shared<lambertian>(color(0.9f, 1.0f, 0.9f, 1.0f));
    auto f_dielectric = std::make_shared<dielectric>(
        1.5f, color(0.9f, 1.0f, 0.9f, 1.0f), 0.01f);

    std::vector<std::shared_ptr<material>> materials = {
        na_metal,      na_lambertian, na_dielectric,
        al_lambertian, f_lambertian,  f_dielectric};

    std::vector<int> material_weights = {3, 2, 1, 3, 1, 2};

    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> pos_dist(-16.0f, 16.0f);
    std::uniform_real_distribution<float> y_dist(-2.0f, 4.0f);
    std::uniform_real_distribution<float> offset_dist(-0.8f, 0.8f);
    std::uniform_real_distribution<float> size_dist(0.4f, 1.2f);

    auto choose_material = [&]() -> std::shared_ptr<material> {
        int total = 0;
        for (int w : material_weights) total += w;
        int r = std::uniform_int_distribution<int>(0, total - 1)(rng);
        int accum = 0;
        for (size_t i = 0; i < materials.size(); ++i) {
            accum += material_weights[i];
            if (r < accum) return materials[i];
        }
        return materials[0];
    };

    for (int i = 0; i < NUM_OBJECTS; ++i) {
        euclidean_coordinate center(pos_dist(rng), y_dist(rng), pos_dist(rng));

        float scale = size_dist(rng);

        auto mat = choose_material();
        world.push_back(std::make_shared<sphere>(center, scale, mat));
    }

    auto ground_mat =
        std::make_shared<lambertian>(color(0.3f, 0.3f, 0.3f, 1.0f));
    world.push_back(
        std::make_shared<plane>(
            euclidean_coordinate(0, -2.5f, 0), euclidean_coordinate(0, 1, 0),
            ground_mat));

    auto point_light1 = std::make_shared<point_light>(
        euclidean_coordinate(8, 10, 5), color(1.0f, 1.0f, 1.0f, 1.0f), 1.0f,
        0.07f, 0.01f);
    auto point_light2 = std::make_shared<point_light>(
        euclidean_coordinate(-5, 8, 6), color(0.6f, 0.5f, 1.0f, 1.0f), 1.0f,
        0.05f, 0.0f);
    auto dir_light = std::make_shared<directional_light>(
        euclidean_coordinate(0.5f, -1.0f, -0.3f),
        color(0.4f, 0.4f, 0.5f, 1.0f));
    lights.push_back(point_light1);
    lights.push_back(point_light2);
    lights.push_back(dir_light);

    bvh_root = std::make_shared<bvh_node>(world, 0, world.size());
}

void handle_input(ExMessage& msg) {
    const float move_speed = 0.2f;
    const float rot_speed = 3.0f;
    switch (msg.message) {
        case WM_KEYDOWN:
            switch (msg.vkcode) {
                case 'W':
                    cam.move_forward(move_speed);
                    break;
                case 'S':
                    cam.move_forward(-move_speed);
                    break;
                case 'A':
                    cam.move_right(-move_speed);
                    break;
                case 'D':
                    cam.move_right(move_speed);
                    break;
                case VK_SPACE:
                    cam.move_up(move_speed);
                    break;
                case VK_SHIFT:
                    cam.move_up(-move_speed);
                    break;
                case 'Q':
                    cam.rotate_roll(rot_speed);
                    break;
                case 'E':
                    cam.rotate_roll(-rot_speed);
                    break;
                case 'I':
                    cam.rotate_pitch(rot_speed);
                    break;
                case 'K':
                    cam.rotate_pitch(-rot_speed);
                    break;
                case 'J':
                    cam.rotate_yaw(rot_speed);
                    break;
                case 'L':
                    cam.rotate_yaw(-rot_speed);
                    break;
                case 'B':
                    use_bvh = !use_bvh;
                    break;
                case 'R':
                    build_scene();
                    break;
            }
            break;
    }
}

int main() {
    omp_set_num_threads(8);

    initgraph(WIDTH, HEIGHT);
    setbkcolor(RGB(0, 0, 0));
    init_random();
    build_scene();
    render();

    auto last_time = std::chrono::steady_clock::now();

    bool running = true;
    ExMessage msg;

    while (running) {
        while (peekmessage(&msg, EM_KEY)) {
            if (msg.message == WM_KEYDOWN && msg.vkcode == VK_ESCAPE) {
                running = false;
                break;
            }
            handle_input(msg);
        }
        render();

        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        if (elapsed >= 1.0f) {
            char title[64];
            snprintf(
                title, 64, "Ray Tracer - SPF: %.2f | BVH: %s", elapsed,
                use_bvh ? "ON" : "OFF");
            SetWindowText(GetHWnd(), title);
            last_time = now;
        }
        Sleep(16);
    }
    closegraph();
    return 0;
}