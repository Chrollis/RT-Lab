#include <camera.h>
#include <graphics.h>
#include <light.h>
#include <object.h>
#include <omp.h>
#include <scene.h>
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

const int WIDTH = 1200;
const int HEIGHT = 900;

int REAL_WIDTH = 400;
int REAL_HEIGHT = 300;
int RENDER_SCALE = 1;

const int MIN_DEPTH = 4;
const float SURVIVAL = 0.8f;
const int MAX_DEPTH = 8;
int NUM_OBJECTS = 100;
const color BACKGROUND_COLOR(0.1f, 0.15f, 0.25f, 1.0f);
int SAMPLES_PER_PIXEL = 4;

char* SAVE_DIRECTORY = nullptr;

accel_t current_accel = accel_t::bvh;
scene sc(current_accel);

camera cam(
    euclidean_coordinate(0, 10, 25),
    euclidean_coordinate(0, 5, 0),
    euclidean_coordinate(0, 1, 0),
    60.0f,
    float(REAL_WIDTH) / REAL_HEIGHT,
    0.1f,
    100.0f);

bool trajectory_mode = false;
float trajectory_angle = 0.0f;
float trajectory_radius = 25.0f;
float trajectory_height = 8.0f;
const float TRAJECTORY_SPEED = 0.5f;

bool is_shadowed(
    const euclidean_coordinate& hit_point,
    const light* lit,
    float light_distance) {
    ray shadow_ray = lit->shadow_ray(hit_point);
    float t_max = (light_distance > 0) ? light_distance : inf;
    return sc.any_hit(shadow_ray, eps, t_max);
}

color ray_color(const ray& r, int depth) {
    if (depth >= MAX_DEPTH) return color::black();

    hit_record rec;
    if (!sc.intersect(r, eps, inf, rec)) {
        float t = 0.5f * (r.direction().y() + 1.0f);
        return BACKGROUND_COLOR * (1.0f - t) +
               color(0.5f, 0.7f, 1.0f, 1.0f) * t;
    }

    euclidean_coordinate view_dir = -r.direction().normalize();

    color direct(0, 0, 0, 1);
    for (const auto& lit : sc.get_lights()) {
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
    float survival_prob = (depth > MIN_DEPTH) ? SURVIVAL : 1.0f;
    if (depth < MAX_DEPTH && random_float() <= survival_prob) {
        ray scattered;
        color attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered)) {
            scattered_color = ray_color(scattered, depth + 1) * attenuation;
        }
    }

    color emitted = rec.mat->emitted(rec);
    return ambient + direct + scattered_color + emitted;
}

IMAGE render(bool draw) {
    std::vector<color> framebuffer(REAL_WIDTH * REAL_HEIGHT);
    float w_inv = 1.0f / REAL_WIDTH;
    float h_inv = 1.0f / REAL_HEIGHT;
    float s_inv = 1.0f / SAMPLES_PER_PIXEL;

#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < REAL_WIDTH * REAL_HEIGHT; ++idx) {
        init_random();
        int x = idx % REAL_WIDTH;
        int y = idx / REAL_WIDTH;
        float pr = 0, pg = 0, pb = 0, pa = 0;
        for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
            float u = (x + random_float()) * w_inv;
            float v = (y + random_float()) * h_inv;
            ray r = cam.get_ray(u, v);
            color rc = ray_color(r, 0);
            pr += rc.r;
            pg += rc.g;
            pb += rc.b;
            pa += rc.a;
        }
        color pixel_color(pr * s_inv, pg * s_inv, pb * s_inv, pa * s_inv);
        framebuffer[idx] = pixel_color.gamma_correct(2.2f);
    }

    IMAGE img(WIDTH, HEIGHT);
    DWORD* bits = GetImageBuffer(&img);
    for (int y = 0; y < HEIGHT; ++y) {
        int src_y = y * REAL_HEIGHT / HEIGHT;
        for (int x = 0; x < WIDTH; ++x) {
            int src_x = x * REAL_WIDTH / WIDTH;
            const color& col = framebuffer[src_y * REAL_WIDTH + src_x];
            bits[y * WIDTH + x] = BGR(col.to_colorref(BLACK));
        }
    }
    if (draw) putimage(0, 0, &img);
    return img;
}

color hsv_to_rgb(float h, float s, float v) {
    h = fmodf(h, 1.0f);
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp, gp, bp;
    if (h < 1.0f / 6.0f) {
        rp = c;
        gp = x;
        bp = 0;
    } else if (h < 2.0f / 6.0f) {
        rp = x;
        gp = c;
        bp = 0;
    } else if (h < 3.0f / 6.0f) {
        rp = 0;
        gp = c;
        bp = x;
    } else if (h < 4.0f / 6.0f) {
        rp = 0;
        gp = x;
        bp = c;
    } else if (h < 5.0f / 6.0f) {
        rp = x;
        gp = 0;
        bp = c;
    } else {
        rp = c;
        gp = 0;
        bp = x;
    }
    return color(rp + m, gp + m, bp + m, 1.0f);
}

void build_scene() {
    sc = scene(current_accel);

    float scene_radius = 12.0f + NUM_OBJECTS / 15.0f;
    float y_max = scene_radius * 0.8f;
    float y_min = -2.5f;

    trajectory_radius = scene_radius * 1.8f;
    trajectory_height = y_max * 1.2f;

    auto white_mat =
        std::make_shared<lambertian>(color(1.0f, 1.0f, 1.0f, 1.0f));
    auto silver_mat =
        std::make_shared<metal>(color(1.0f, 1.0f, 1.0f, 1.0f), 0.05f);
    auto glass_mat = std::make_shared<dielectric>(
        1.46f, color(0.9f, 1.0f, 0.9f, 1.0f), 0.01f);

    std::vector<std::shared_ptr<material>> materials = {
        white_mat, silver_mat, glass_mat};

    for (int i = 0; i < NUM_OBJECTS; ++i) {
        float r = random_float() * scene_radius;
        float theta = random_float() * 2 * pi;
        float x = r * cos(theta);
        float z = r * sin(theta);
        float y = random_float(y_min, y_max);
        euclidean_coordinate center(x, y, z);
        float radius = random_float(0.4f, 1.2f);
        int mat_idx = 0;
        if (random_float() < 0.3f)
            mat_idx = 1;
        else if (random_float() > 0.8f)
            mat_idx = 2;
        sc.add_object(
            std::make_shared<sphere>(center, radius, materials[mat_idx]));
    }

    auto ground = std::make_shared<lambertian>(color(0.4f, 0.4f, 0.45f, 1.0f));
    sc.add_plane(
        std::make_shared<plane>(
            euclidean_coordinate(0, y_min, 0), euclidean_coordinate(0, 1, 0),
            ground));

    auto dl = std::make_shared<directional_light>(
        euclidean_coordinate(0.5f, -1.0f, -0.3f).normalize(),
        color(0.8f, 0.9f, 1.0f, 1.0f) * 1.2f);

    float light_x = scene_radius * 0.6f;
    float light_z = scene_radius * 0.5f;
    float light_y = y_max * 1.1f;
    auto pl = std::make_shared<point_light>(
        euclidean_coordinate(light_x, light_y, light_z),
        color(1.0f, 0.85f, 0.7f, 1.0f), 1.0f, 0.04f, 0.01f);

    sc.add_light(dl);
    sc.add_light(pl);

    sc.rebuild_accel();
}

void handle_input(ExMessage& msg) {
    const float move_speed = 0.2f;
    const float rot_speed = 3.0f;

    if (trajectory_mode && msg.message == WM_KEYDOWN) {
        if (!(msg.vkcode == 'T' || msg.vkcode == VK_ESCAPE)) return;
    }

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
                    current_accel = accel_t::bvh;
                    sc.set_accel_type(accel_t::bvh);
                    break;
                case 'N':
                    current_accel = accel_t::linear;
                    sc.set_accel_type(accel_t::linear);
                    break;
                case 'M':
                    current_accel = accel_t::uniform_grid;
                    sc.set_accel_type(accel_t::uniform_grid);
                    break;
                case 'R':
                    build_scene();
                    if (!trajectory_mode) {
                        float y_min = -2.5f;
                        float y_max = (12.0f + NUM_OBJECTS / 15.0f) * 0.8f;
                        float lookat_y = (y_max + y_min) * 0.5f;
                        cam.set_pose(
                            euclidean_coordinate(
                                0, trajectory_height, trajectory_radius),
                            euclidean_coordinate(0, lookat_y, 0));
                    }
                    break;
                case 'T':
                    trajectory_mode = !trajectory_mode;
                    if (trajectory_mode) {
                        trajectory_angle = 0.0f;
                        float x = trajectory_radius * cos(trajectory_angle);
                        float z = trajectory_radius * sin(trajectory_angle);
                        cam.set_pose(
                            euclidean_coordinate(x, trajectory_height, z),
                            euclidean_coordinate(0, 5, 0));
                    }
                    break;
            }
            break;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        RENDER_SCALE = std::atoi(argv[1]);
        if (RENDER_SCALE < 1) RENDER_SCALE = 1;
        REAL_WIDTH = WIDTH / RENDER_SCALE;
        REAL_HEIGHT = HEIGHT / RENDER_SCALE;
        if (REAL_WIDTH < 1) REAL_WIDTH = 1;
        if (REAL_HEIGHT < 1) REAL_HEIGHT = 1;
    }
    if (argc >= 3) {
        NUM_OBJECTS = std::atoi(argv[2]);
        if (NUM_OBJECTS < 1) NUM_OBJECTS = 1;
    }
    if (argc >= 4) {
        SAMPLES_PER_PIXEL = std::atoi(argv[3]);
        if (SAMPLES_PER_PIXEL < 1) SAMPLES_PER_PIXEL = 1;
    }
    if (argc >= 5) {
        SAVE_DIRECTORY = argv[4];
        if (SAVE_DIRECTORY) {
            DWORD attrib = GetFileAttributes(SAVE_DIRECTORY);
            if (attrib == INVALID_FILE_ATTRIBUTES ||
                !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
                if (!CreateDirectory(SAVE_DIRECTORY, NULL)) {
                    SAVE_DIRECTORY = nullptr;
                }
            }
        }
    }
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    cam = camera(
        euclidean_coordinate(0, 10, 25), euclidean_coordinate(0, 5, 0),
        euclidean_coordinate(0, 1, 0), 60.0f, float(REAL_WIDTH) / REAL_HEIGHT,
        0.1f, 100.0f);

    omp_set_num_threads(omp_get_max_threads());
    init_random();
    build_scene();

    if (!trajectory_mode) {
        float y_min = -2.5f;
        float y_max = (12.0f + NUM_OBJECTS / 15.0f) * 0.8f;
        float lookat_y = (y_max + y_min) * 0.5f;
        cam = camera(
            euclidean_coordinate(0, trajectory_height, trajectory_radius),
            euclidean_coordinate(0, lookat_y, 0), euclidean_coordinate(0, 1, 0),
            60.0f, float(REAL_WIDTH) / REAL_HEIGHT, 0.1f, 100.0f);
    }

    if (SAVE_DIRECTORY != nullptr) {
        initgraph(200, 150, EX_SHOWCONSOLE);
        HWND hWnd = GetHWnd();
        ShowWindow(hWnd, SW_HIDE);
        IMAGE offscreen(WIDTH, HEIGHT);
        int FPS = 24;
        if (argc >= 6) {
            FPS = std::atoi(argv[5]);
            if (FPS < 1) FPS = 1;
        }
        const int TOTAL_FRAMES = FPS * 15;
        float angle_step = 2.0f * pi / TOTAL_FRAMES;
        float current_angle = 0.0f;
        sc.set_accel_type(accel_t::uniform_grid);

        printf(
            "Offline rendering %d frames to %s\n", TOTAL_FRAMES,
            SAVE_DIRECTORY);
        for (int frame = 1; frame <= TOTAL_FRAMES; ++frame) {
            float x = trajectory_radius * cos(current_angle);
            float z = trajectory_radius * sin(current_angle);
            cam.set_pose(
                euclidean_coordinate(x, trajectory_height, z),
                euclidean_coordinate(0, 5, 0));
            offscreen = std::move(render(false));
            char filename[256];
            snprintf(
                filename, sizeof(filename), "%s\\frame_%03d.bmp",
                SAVE_DIRECTORY, frame);
            saveimage(filename, &offscreen);
            printf(
                "Saved %s (%.1f%%)\n", filename, 100.0f * frame / TOTAL_FRAMES);
            current_angle += angle_step;
        }
        SetWorkingImage(NULL);
        closegraph();
        return 0;
    }

    initgraph(WIDTH, HEIGHT);
    setbkcolor(RGB(0, 0, 0));

    auto last_time = std::chrono::steady_clock::now();
    bool running = true;
    ExMessage msg;
    int frame_count = 0;

    while (running) {
        while (peekmessage(&msg, EM_KEY)) {
            if (msg.message == WM_KEYDOWN && msg.vkcode == VK_ESCAPE) {
                running = false;
                break;
            }
            handle_input(msg);
        }

        if (trajectory_mode) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            if (dt > 0.1f) dt = 0.1f;
            trajectory_angle += TRAJECTORY_SPEED * dt;
            if (trajectory_angle > 2 * pi) trajectory_angle -= 2 * pi;

            float x = trajectory_radius * cos(trajectory_angle);
            float z = trajectory_radius * sin(trajectory_angle);
            cam.set_pose(
                euclidean_coordinate(x, trajectory_height, z),
                euclidean_coordinate(0, 5, 0));
        }

        render(true);
        frame_count++;

        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        const char* accel_name = "Linear";
        if (current_accel == accel_t::bvh)
            accel_name = "BVH";
        else if (current_accel == accel_t::uniform_grid)
            accel_name = "Uniform Grid";
        char title[128];
        snprintf(
            title, sizeof(title),
            "RayTracer | SPF: %.2f | Accel: %s | Scale: %d", elapsed,
            accel_name, RENDER_SCALE);
        SetWindowText(GetHWnd(), title);
        last_time = now;
    }
    closegraph();
    return 0;
}