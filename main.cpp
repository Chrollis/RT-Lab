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
#include <cstring>
#include <memory>
#include <random>
#include <vector>
#undef max
#undef min

using namespace chrray;

int WIDTH = 960;
int HEIGHT = 720;

int NUM_OBJECTS = 100;
int actual_objects = 0;
int MIN_DEPTH = 4;
int MAX_DEPTH = 8;
int SAMPLES_PER_PIXEL = 4;

int offline_fps = 24;
bool auto_trajectory = false;

const float SURVIVAL = 0.8f;
const color BACKGROUND_COLOR(0.1f, 0.15f, 0.25f, 1.0f);

char* SAVE_DIRECTORY = nullptr;

accel_t current_accel = accel_t::bvh;
scene sc(current_accel);

camera cam(
    euclidean_coordinate(0, 10, 25),
    euclidean_coordinate(0, 5, 0),
    euclidean_coordinate(0, 1, 0),
    60.0f,
    float(WIDTH) / HEIGHT,
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
    color throughput(1.0f, 1.0f, 1.0f, 1.0f);
    color accumulated(0.0f, 0.0f, 0.0f, 1.0f);
    ray current_ray = r;
    int current_depth = 0;

    while (current_depth < MAX_DEPTH) {
        hit_record rec;
        if (!sc.intersect(current_ray, eps, inf, rec)) {
            float t = 0.5f * (current_ray.direction().y() + 1.0f);
            color bg = BACKGROUND_COLOR * (1.0f - t) +
                       color(0.5f, 0.7f, 1.0f, 1.0f) * t;
            accumulated = accumulated + throughput * bg;
            break;
        }

        euclidean_coordinate view_dir = -current_ray.direction().normalize();

        color direct(0.0f, 0.0f, 0.0f, 1.0f);
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
            color spec = rec.mat->specular_color(rec, view_dir) *
                         light_intensity * spec_intensity;

            direct = direct + diff + spec;
        }

        color ambient = rec.mat->diffuse_color(rec, view_dir) * 0.1f;
        accumulated = accumulated + throughput * (ambient + direct);

        float survival_prob = (current_depth > MIN_DEPTH) ? SURVIVAL : 1.0f;
        if (random_float() > survival_prob) {
            break;
        }

        ray scattered;
        color attenuation;
        if (!rec.mat->scatter(current_ray, rec, attenuation, scattered)) {
            break;
        }

        throughput = throughput * attenuation * (1.0f / survival_prob);
        current_ray = scattered;
        current_depth++;
    }

    return accumulated;
}

IMAGE render(bool draw) {
    std::vector<color> framebuffer(WIDTH * HEIGHT);
    float w_inv = 1.0f / WIDTH;
    float h_inv = 1.0f / HEIGHT;
    float s_inv = 1.0f / SAMPLES_PER_PIXEL;

#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < WIDTH * HEIGHT; ++idx) {
        init_random();
        int x = idx % WIDTH;
        int y = idx / WIDTH;
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
        for (int x = 0; x < WIDTH; ++x) {
            const color& col = framebuffer[y * WIDTH + x];
            bits[y * WIDTH + x] = BGR(col.to_colorref(BLACK));
        }
    }
    if (draw) putimage(0, 0, &img);
    return img;
}

void build_scene() {
    sc = scene(current_accel);

    float scene_radius = 10.0f + NUM_OBJECTS / 50.0f;
    float y_max = scene_radius + 2.0f;
    float y_min = 1.0f;

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

    const int MAX_ATTEMPTS = 500;
    std::vector<euclidean_coordinate> centers;
    std::vector<float> radii;

    actual_objects = 0;
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            float r = random_float() * scene_radius;
            float theta = random_float() * 2 * pi;
            float phi = random_float() * pi * 0.5;
            float x = r * cosf(phi) * cosf(theta);
            float z = r * cosf(phi) * sinf(theta);
            float y = r * sinf(phi) + y_min;
            float radius = random_float(0.5f, y_min);
            euclidean_coordinate center(x, y, z);

            bool overlap = false;
            for (size_t j = 0; j < centers.size(); ++j) {
                float dist = (center - centers[j]).length();
                float min_dist = radius + radii[j];
                if (dist < min_dist) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                centers.push_back(center);
                radii.push_back(radius);
                int mat_idx = 0;
                if (random_float() < 0.3f)
                    mat_idx = 1;
                else if (random_float() > 0.8f)
                    mat_idx = 2;
                sc.add_object(
                    std::make_shared<sphere>(
                        center, radius, materials[mat_idx]));
                actual_objects++;
                break;
            }
        }
    }

    auto ground = std::make_shared<lambertian>(color(0.4f, 0.4f, 0.45f, 1.0f));
    sc.add_plane(
        std::make_shared<plane>(
            euclidean_coordinate(0, 0, 0), euclidean_coordinate(0, 1, 0),
            ground));

    auto dl = std::make_shared<directional_light>(
        euclidean_coordinate(0.5f, -1.0f, -0.5f).normalize(),
        color(0.8f, 0.9f, 1.0f, 1.0f) * 1.2f);

    auto pl1 = std::make_shared<point_light>(
        euclidean_coordinate(
            scene_radius * 0.5f, y_max * 1.2f, scene_radius * 0.5f),
        color(1.0f, 0.85f, 0.7f, 1.0f), 1.0f, 0.01f, 0.01f);
    auto pl2 = std::make_shared<point_light>(
        euclidean_coordinate(
            -scene_radius * 0.5f, y_max * 1.2f, -scene_radius * 0.5f),
        color(1.0f, 0.85f, 0.7f, 1.0f), 1.0f, 0.01f, 0.01f);

    sc.add_light(dl);
    sc.add_light(pl1);
    sc.add_light(pl2);

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

void parse_command_line(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            initgraph(WIDTH, HEIGHT, EX_SHOWCONSOLE);
            HWND hWnd = GetHWnd();
            ShowWindow(hWnd, SW_HIDE);
            printf(
                "Usage: raytracer [options]\n"
                "Options:\n"
                "  -h                     Show this help message\n"
                "  -q <w> <h> [fps]       Set resolution to w x h (fps default "
                "24)\n"
                "  -c <count>             Set number of objects\n"
                "  -r <min> <max>         Set recursion depth min and max\n"
                "  -a <samples>           Set MSAA samples per pixel\n"
                "  -ql                    Set resolution to 640x480, offline "
                "fps=6\n"
                "  -qm                    Set resolution to 960x720, offline "
                "fps=12\n"
                "  -qh                    Set resolution to 1440x1080, offline "
                "fps=24\n"
                "  -cf                    Set object count = 10\n"
                "  -cn                    Set object count = 100\n"
                "  -cm                    Set object count = 500\n"
                "  -rd                    Set recursion depth: MIN=16, MAX=32\n"
                "  -rm                    Set recursion depth: MIN=8, MAX=16\n"
                "  -rs                    Set recursion depth: MIN=4, MAX=8\n"
                "  -al                    Set MSAA samples = 2\n"
                "  -am                    Set MSAA samples = 4\n"
                "  -ah                    Set MSAA samples = 8\n"
                "  -s <path>              Set save directory for offline "
                "rendering\n"
                "  -l                     Set initial rendering algorithm to "
                "Linear\n"
                "  -b                     Set initial rendering algorithm to "
                "BVH\n"
                "  -u                     Set initial rendering algorithm to "
                "Uniform Grid\n"
                "  -t                     Auto trajectory mode (real-time) "
                "or show frames in offline\n");
            system("pause");
            closegraph();
            exit(0);
        } else if (strcmp(argv[i], "-q") == 0) {
            if (i + 2 >= argc) {
                printf("Error: -q requires width and height.\n");
                exit(1);
            }
            WIDTH = atoi(argv[++i]);
            HEIGHT = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                offline_fps = atoi(argv[++i]);
            } else {
                offline_fps = 24;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc) {
                printf("Error: -c requires a count.\n");
                exit(1);
            }
            NUM_OBJECTS = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            if (i + 2 >= argc) {
                printf("Error: -r requires min_depth and max_depth.\n");
                exit(1);
            }
            MIN_DEPTH = atoi(argv[++i]);
            MAX_DEPTH = atoi(argv[++i]);
            if (MIN_DEPTH > MAX_DEPTH) {
                printf("Error: MIN_DEPTH must be <= MAX_DEPTH\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "-a") == 0) {
            if (i + 1 >= argc) {
                printf("Error: -a requires a samples count.\n");
                exit(1);
            }
            SAMPLES_PER_PIXEL = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ql") == 0) {
            WIDTH = 640;
            HEIGHT = 480;
            offline_fps = 6;
        } else if (strcmp(argv[i], "-qm") == 0) {
            WIDTH = 960;
            HEIGHT = 720;
            offline_fps = 12;
        } else if (strcmp(argv[i], "-qh") == 0) {
            WIDTH = 1440;
            HEIGHT = 1080;
            offline_fps = 24;
        } else if (strcmp(argv[i], "-cf") == 0) {
            NUM_OBJECTS = 10;
        } else if (strcmp(argv[i], "-cn") == 0) {
            NUM_OBJECTS = 100;
        } else if (strcmp(argv[i], "-cm") == 0) {
            NUM_OBJECTS = 500;
        } else if (strcmp(argv[i], "-rd") == 0) {
            MIN_DEPTH = 16;
            MAX_DEPTH = 32;
        } else if (strcmp(argv[i], "-rm") == 0) {
            MIN_DEPTH = 8;
            MAX_DEPTH = 16;
        } else if (strcmp(argv[i], "-rs") == 0) {
            MIN_DEPTH = 4;
            MAX_DEPTH = 8;
        } else if (strcmp(argv[i], "-al") == 0) {
            SAMPLES_PER_PIXEL = 2;
        } else if (strcmp(argv[i], "-am") == 0) {
            SAMPLES_PER_PIXEL = 4;
        } else if (strcmp(argv[i], "-ah") == 0) {
            SAMPLES_PER_PIXEL = 8;
        } else if (strcmp(argv[i], "-s") == 0) {
            if (i + 1 >= argc) {
                printf("Error: -s requires a directory path.\n");
                exit(1);
            }
            SAVE_DIRECTORY = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0) {
            current_accel = accel_t::linear;
            sc.set_accel_type(current_accel);
        } else if (strcmp(argv[i], "-b") == 0) {
            current_accel = accel_t::bvh;
            sc.set_accel_type(current_accel);
        } else if (strcmp(argv[i], "-u") == 0) {
            current_accel = accel_t::uniform_grid;
            sc.set_accel_type(current_accel);
        } else if (strcmp(argv[i], "-t") == 0) {
            auto_trajectory = true;
        }
    }
}

int main(int argc, char** argv) {
    parse_command_line(argc, argv);

    if (SAVE_DIRECTORY) {
        DWORD attrib = GetFileAttributes(SAVE_DIRECTORY);
        if (attrib == INVALID_FILE_ATTRIBUTES ||
            !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
            if (!CreateDirectory(SAVE_DIRECTORY, NULL)) {
                SAVE_DIRECTORY = nullptr;
            }
        }
    }

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

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
            60.0f, float(WIDTH) / HEIGHT, 0.1f, 100.0f);
    }

    if (SAVE_DIRECTORY != nullptr) {
        initgraph(WIDTH, HEIGHT, EX_SHOWCONSOLE);
        HWND hWnd = GetHWnd();
        if (!auto_trajectory) {
            ShowWindow(hWnd, SW_HIDE);
        } else {
            SetWindowText(
                hWnd, "RayTracer - Offline Rendering (Auto Trajectory)");
        }
        IMAGE offscreen(WIDTH, HEIGHT);
        const int TOTAL_FRAMES = offline_fps * 15;
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

            offscreen = std::move(render(auto_trajectory));
            char filename[256];
            snprintf(
                filename, sizeof(filename), "%s\\frame_%03d.bmp",
                SAVE_DIRECTORY, frame);
            saveimage(filename, &offscreen);
            printf(
                "Saved %s (%.1f%%)\n", filename, 100.0f * frame / TOTAL_FRAMES);
            current_angle += angle_step;
        }
        system("pause");
        closegraph();
        return 0;
    }

    initgraph(WIDTH, HEIGHT);
    setbkcolor(RGB(0, 0, 0));

    auto last_time = std::chrono::steady_clock::now();
    bool running = true;
    ExMessage msg;
    int frame_count = 0;
    if (auto_trajectory) {
        trajectory_mode = true;
        trajectory_angle = 0.0f;
        float x = trajectory_radius * cos(trajectory_angle);
        float z = trajectory_radius * sin(trajectory_angle);
        cam.set_pose(
            euclidean_coordinate(x, trajectory_height, z),
            euclidean_coordinate(0, 5, 0));
    }

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
            "RayTracer | OBJ: %d | SPF: %.2f | Accel: %s | %dx%d",
            actual_objects, elapsed, accel_name, WIDTH, HEIGHT);
        SetWindowText(GetHWnd(), title);
        last_time = now;
    }
    closegraph();
    return 0;
}