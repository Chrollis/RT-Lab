#include <omp.h>
#include <renderer.h>
#include <texture.h>
#include <utils.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace chrray {
static std::string format_duration(float seconds) {
    int total_ms = static_cast<int>(seconds * 1000.0f);
    int ms = total_ms % 1000;
    int total_sec = total_ms / 1000;
    int sec = total_sec % 60;
    int total_min = total_sec / 60;
    int min = total_min % 60;
    int total_hour = total_min / 60;
    int hour = total_hour % 24;
    int day = total_hour / 24;

    std::ostringstream oss;
    if (day > 0) {
        oss << day << "d " << hour << "h " << min << "m " << sec << "s " << ms
            << "ms";
    } else if (hour > 0) {
        oss << hour << "h " << min << "m " << sec << "s " << ms << "ms";
    } else if (min > 0) {
        oss << min << "m " << sec << "s " << ms << "ms";
    } else if (sec > 0) {
        oss << sec << "s " << ms << "ms";
    } else {
        oss << ms << "ms";
    }
    return oss.str();
}

static bool is_shadowed(
    const euclidean_coordinate& hit_point,
    const light* lit,
    float light_distance,
    const scene& sc) {
    ray shadow_ray = lit->shadow_ray(hit_point);
    float t_max = (light_distance > 0) ? light_distance : inf;
    return sc.any_hit(shadow_ray, eps, t_max);
}

renderer::renderer()
    : opt_("RT-Lab", "Ray Tracing Renderer"),
      sc_(accel_type_),
      cam_(
          euclidean_coordinate(0, 10, 25),
          euclidean_coordinate(0, 5, 0),
          euclidean_coordinate(0, 1, 0),
          60.0f,
          float(width_) / height_,
          0.1f,
          100.0f) {
    opt_.add_options()("h,help", "Show this help message")

        ("q,resolution", "Set resolution (WxH[:FPS])",
         cxxopts::value<std::string>())(
            "preset", "Preset resolution (low|mid|high)",
            cxxopts::value<std::string>())(
            "c,objects", "Number of objects (N)", cxxopts::value<int>())(
            "r,recursion", "Recursion depth (MIN:MAX)",
            cxxopts::value<std::string>())(
            "a,samples", "MSAA samples per pixel (N)", cxxopts::value<int>())(
            "s,save-dir", "Save directory (DIR)",
            cxxopts::value<std::string>())(
            "S,save-single", "Save single frame and exit")(
            "A,accel", "Accelerator (linear|bvh|uniform)",
            cxxopts::value<std::string>())(
            "t,trajectory", "Auto trajectory mode")(
            "e,seed", "Random seed (SEED, 0=auto)",
            cxxopts::value<unsigned int>())(
            "shadows", "Enable shadows (default)")(
            "no-shadows", "Disable shadows")(
            "T,threads", "OpenMP threads (N)", cxxopts::value<int>())(
            "R,rasterize", "Rasterized preview mode")(
            "O,obj-type", "Object type (sphere|triangle|mixed:0.5)",
            cxxopts::value<std::string>())(
            "M,mesh", "Load mesh file (FILE)", cxxopts::value<std::string>());
}

void renderer::error_and_exit(const std::string& msg) const {
    std::cerr << "Error: " << msg << std::endl;
    print_help_and_exit(1);
}

static std::shared_ptr<material> choose_material(int i = -1) {
    static auto white_mat =
        std::make_shared<lambertian>(color(1.0f, 1.0f, 1.0f, 1.0f));
    static auto silver_mat =
        std::make_shared<metal>(color(1.0f, 1.0f, 1.0f, 1.0f), 0.05f);
    static auto glass_mat = std::make_shared<dielectric>(
        1.46f, color(0.9f, 1.0f, 0.9f, 1.0f), 0.01f);

    if (i < 0) {
        float r = random_float();
        if (r < 0.333f)
            return white_mat;
        else if (r < 0.667f)
            return silver_mat;
        else
            return glass_mat;
    } else {
        if (i % 3 == 0)
            return white_mat;
        else if (i % 3 == 1)
            return silver_mat;
        else
            return glass_mat;
    }
}

void renderer::build_scene() {
    sc_ = scene(accel_type_);

    float scene_radius = 10.0f + num_objects_ / 50.0f;
    float y_max = scene_radius + 2.0f;
    float y_min = 1.0f;
    trajectory_radius_ = scene_radius * 1.8f;
    trajectory_height_ = y_max * 1.2f;

    actual_objects_ = 0;

    if (obj_type_ == object_t::mesh) {
        sc_.add_object(
            std::make_shared<mesh>(
                mesh_filename_, scene_radius, y_min, choose_material(0)));
        actual_objects_ = 1;
    } else {
        int sphere_count = 0, triangle_count = 0;
        switch (obj_type_) {
            case object_t::sphere:
                sphere_count = num_objects_;
                break;
            case object_t::triangle:
                triangle_count = num_objects_;
                break;
            case object_t::mixed:
                sphere_count = (int)(num_objects_ * sphere_ratio_);
                triangle_count = num_objects_ - sphere_count;
                break;
            default:
                break;
        }
        std::vector<euclidean_coordinate> centers;
        std::vector<float> radii;

        for (int i = 0; i < sphere_count; ++i) {
            bool placed = false;
            for (int attempt = 0; attempt < 500; ++attempt) {
                float r = random_float() * scene_radius;
                float theta = random_float() * 2 * pi;
                float phi = random_float() * pi * 0.5f;
                float x = r * cosf(phi) * cosf(theta);
                float z = r * cosf(phi) * sinf(theta);
                float y = r * sinf(phi) + y_min;
                float radius = random_float(0.5f, y_min);
                euclidean_coordinate center(x, y, z);

                bool overlap = false;
                for (size_t j = 0; j < centers.size(); ++j) {
                    float dist = (center - centers[j]).length();
                    if (dist < radius + radii[j]) {
                        overlap = true;
                        break;
                    }
                }
                if (!overlap) {
                    centers.push_back(center);
                    radii.push_back(radius);
                    sc_.add_object(
                        std::make_shared<sphere>(
                            center, radius, choose_material()));
                    actual_objects_++;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                float r = random_float() * scene_radius;
                float theta = random_float() * 2 * pi;
                float phi = random_float() * pi * 0.5f;
                float x = r * cosf(phi) * cosf(theta);
                float z = r * cosf(phi) * sinf(theta);
                float y = r * sinf(phi) + y_min;
                float radius = random_float(0.5f, y_min);
                sc_.add_object(
                    std::make_shared<sphere>(
                        euclidean_coordinate(x, y, z), radius,
                        choose_material()));
                actual_objects_++;
            }
        }

        for (int i = 0; i < triangle_count; ++i) {
            euclidean_coordinate v0, v1, v2;
            bool valid = false;
            int attempts = 0;
            while (!valid && attempts < 100) {
                v0 = euclidean_coordinate(
                    random_float(-scene_radius, scene_radius),
                    random_float(y_min, y_max),
                    random_float(-scene_radius, scene_radius));
                v1 = v0 + euclidean_coordinate(
                              random_float(-1, 1), random_float(-1, 1),
                              random_float(-1, 1)) *
                              0.5f;
                v2 = v0 + euclidean_coordinate(
                              random_float(-1, 1), random_float(-1, 1),
                              random_float(-1, 1)) *
                              0.5f;
                if ((v1 - v0).length() > 1e-4f && (v2 - v0).length() > 1e-4f)
                    valid = true;
                attempts++;
            }
            if (valid) {
                sc_.add_object(
                    std::make_shared<triangle>(v0, v1, v2, choose_material()));
                actual_objects_++;
            }
        }
    }

    auto ground = std::make_shared<metal>(color(0.4f, 0.4f, 0.45f, 1.0f), 0.0f);
    sc_.add_plane(
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

    sc_.add_light(dl);
    sc_.add_light(pl1);
    sc_.add_light(pl2);

    sc_.rebuild_accel();
    sc_.build_triangles();
}

color renderer::ray_color(const ray& r, int depth) const {
    color throughput(1.0f, 1.0f, 1.0f, 1.0f);
    color accumulated(0.0f, 0.0f, 0.0f, 1.0f);
    ray current_ray = r;
    int current_depth = 0;

    while (current_depth < max_depth_) {
        hit_record rec;
        if (!sc_.intersect(current_ray, eps, inf, rec)) {
            float t = 0.5f * (current_ray.direction().y() + 1.0f);
            color bg = color(0.1f, 0.15f, 0.25f, 1.0f) * (1.0f - t) +
                       color(0.5f, 0.7f, 1.0f, 1.0f) * t;
            accumulated = accumulated + throughput * bg;
            break;
        }

        euclidean_coordinate view_dir = -current_ray.direction().normalize();

        color direct(0.0f, 0.0f, 0.0f, 1.0f);
        for (const auto& lit : sc_.get_lights()) {
            euclidean_coordinate light_dir = lit->direction(rec.p);
            float light_dist = lit->distance(rec.p);
            if (enable_shadows_ &&
                is_shadowed(rec.p, lit.get(), light_dist, sc_))
                continue;

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

        float survival_prob = (current_depth > min_depth_) ? 0.8f : 1.0f;
        if (random_float() > survival_prob) break;

        ray scattered;
        color attenuation;
        if (!rec.mat->scatter(current_ray, rec, attenuation, scattered)) break;

        throughput = throughput * attenuation * (1.0f / survival_prob);
        current_ray = scattered;
        current_depth++;
    }

    return accumulated;
}

IMAGE renderer::render(bool draw) {
    std::vector<color> framebuffer(width_ * height_);
    float w_inv = 1.0f / width_;
    float h_inv = 1.0f / height_;
    float s_inv = 1.0f / samples_per_pixel_;

#pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < width_ * height_; ++idx) {
        init_random(0);
        int x = idx % width_;
        int y = idx / width_;
        float pr = 0, pg = 0, pb = 0, pa = 0;
        for (int s = 0; s < samples_per_pixel_; ++s) {
            float u = (x + random_float()) * w_inv;
            float v = (y + random_float()) * h_inv;
            ray r = cam_.get_ray(u, v);
            color rc = ray_color(r, 0);
            pr += rc.r;
            pg += rc.g;
            pb += rc.b;
            pa += rc.a;
        }
        color pixel_color(pr * s_inv, pg * s_inv, pb * s_inv, pa * s_inv);
        framebuffer[idx] = pixel_color.gamma_correct(2.2f);
    }

    IMAGE img(width_, height_);
    DWORD* bits = GetImageBuffer(&img);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const color& col = framebuffer[y * width_ + x];
            bits[y * width_ + x] = BGR(col.to_colorref(BLACK));
        }
    }
    if (draw) putimage(0, 0, &img);
    return img;
}

IMAGE renderer::render_rasterized(bool draw) {
    const int w = width_, h = height_;
    const size_t pixel_count = w * h;

    const auto& triangles = sc_.get_global_triangles();
    if (triangles.empty()) return IMAGE(w, h);

    homogeneous_transform V = cam_.view_matrix();
    homogeneous_transform P = cam_.projection_matrix();
    homogeneous_transform VP = P * V;

    euclidean_coordinate light_dir(0.5f, -1.0f, -0.5f);
    light_dir = light_dir.normalize();

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> local_zbuffers(num_threads);
    std::vector<std::vector<color>> local_framebuffers(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        local_zbuffers[t].assign(pixel_count, INFINITY);
        local_framebuffers[t].assign(pixel_count, color::black());
    }

#pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < (int)triangles.size(); ++idx) {
        int tid = omp_get_thread_num();
        auto& zbuffer = local_zbuffers[tid];
        auto& framebuffer = local_framebuffers[tid];

        const auto& tri = triangles[idx];
        euclidean_coordinate v0_w = tri.v0, v1_w = tri.v1, v2_w = tri.v2;
        euclidean_coordinate center = (v0_w + v1_w + v2_w) / 3.0f;
        euclidean_coordinate view_dir = (cam_.origin() - center).normalize();
        if (tri.normal.dot(view_dir) >= 0) continue;

        homogeneous_coordinate h0 = v0_w.homogenize();
        homogeneous_coordinate h1 = v1_w.homogenize();
        homogeneous_coordinate h2 = v2_w.homogenize();
        h0 = VP * h0;
        h1 = VP * h1;
        h2 = VP * h2;

        float w0 = h0.w(), w1 = h1.w(), w2 = h2.w();
        if (w0 <= 0 && w1 <= 0 && w2 <= 0) continue;

        euclidean_coordinate ndc0 = h0.dehomogenize();
        euclidean_coordinate ndc1 = h1.dehomogenize();
        euclidean_coordinate ndc2 = h2.dehomogenize();

        if (fabs(ndc0.x()) > 2.0f || fabs(ndc0.y()) > 2.0f ||
            fabs(ndc0.z()) > 2.0f || fabs(ndc1.x()) > 2.0f ||
            fabs(ndc1.y()) > 2.0f || fabs(ndc1.z()) > 2.0f ||
            fabs(ndc2.x()) > 2.0f || fabs(ndc2.y()) > 2.0f ||
            fabs(ndc2.z()) > 2.0f)
            continue;

        auto to_screen =
            [&](const euclidean_coordinate& ndc) -> euclidean_coordinate {
            return euclidean_coordinate(
                (ndc.x() + 1.0f) * 0.5f * w, (1.0f - ndc.y()) * 0.5f * h,
                ndc.z());
        };
        euclidean_coordinate s0 = to_screen(ndc0);
        euclidean_coordinate s1 = to_screen(ndc1);
        euclidean_coordinate s2 = to_screen(ndc2);

        if (!std::isfinite(s0.x()) || !std::isfinite(s0.y()) ||
            !std::isfinite(s1.x()) || !std::isfinite(s1.y()) ||
            !std::isfinite(s2.x()) || !std::isfinite(s2.y()))
            continue;

        int xmin =
            std::max(0, (int)std::floor(std::min({s0.x(), s1.x(), s2.x()})));
        int xmax =
            std::min(w - 1, (int)std::ceil(std::max({s0.x(), s1.x(), s2.x()})));
        int ymin =
            std::max(0, (int)std::floor(std::min({s0.y(), s1.y(), s2.y()})));
        int ymax =
            std::min(h - 1, (int)std::ceil(std::max({s0.y(), s1.y(), s2.y()})));
        if (xmin > xmax || ymin > ymax) continue;

        color base_color =
            tri.mat->diffuse_color(hit_record(), euclidean_coordinate(0, 0, 0));

        auto edge = [](const euclidean_coordinate& a,
                       const euclidean_coordinate& b,
                       const euclidean_coordinate& p) -> float {
            return (b.x() - a.x()) * (p.y() - a.y()) -
                   (b.y() - a.y()) * (p.x() - a.x());
        };
        float area = edge(s0, s1, s2);
        if (area == 0.0f) continue;
        float inv_area = 1.0f / area;
        euclidean_coordinate n = tri.normal.normalize();

        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                euclidean_coordinate p(x + 0.5f, y + 0.5f, 0);
                float w0 = edge(s1, s2, p);
                float w1 = edge(s2, s0, p);
                float w2 = edge(s0, s1, p);
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                w0 *= inv_area;
                w1 *= inv_area;
                w2 *= inv_area;
                float depth = w0 * s0.z() + w1 * s1.z() + w2 * s2.z();
                int pixel_idx = y * w + x;
                if (depth < zbuffer[pixel_idx]) {
                    zbuffer[pixel_idx] = depth;
                    float ndotl = std::max(0.0f, n.dot(light_dir));
                    color lit = base_color * ndotl + base_color * 0.1f;
                    framebuffer[pixel_idx] = lit.gamma_correct(2.2f);
                }
            }
        }
    }

    IMAGE img(w, h);
    DWORD* img_bits = GetImageBuffer(&img);
    memset(img_bits, 0, w * h * sizeof(DWORD));

    for (size_t idx = 0; idx < pixel_count; ++idx) {
        float best_depth = INFINITY;
        color best_color = color::black();
        for (int t = 0; t < num_threads; ++t) {
            float d = local_zbuffers[t][idx];
            if (d < best_depth) {
                best_depth = d;
                best_color = local_framebuffers[t][idx];
            }
        }
        img_bits[idx] = best_color.to_colorref(BLACK);
    }

    if (draw) putimage(0, 0, &img);
    return img;
}

static void update_progress(
    int frame, int total_frames, float dt, const std::string& filename) {
    float progress = static_cast<float>(frame) / total_frames;
    int percent = static_cast<int>(progress * 100.0f);

    const int bar_width = 50;
    int filled = static_cast<int>(progress * bar_width);
    std::string bar = "[" + std::string(filled, '#') +
                      std::string(bar_width - filled, '-') + "]";

    float remaining_sec = dt * (total_frames - frame);
    int remain_h = static_cast<int>(remaining_sec) / 3600;
    int remain_m = (static_cast<int>(remaining_sec) % 3600) / 60;
    int remain_s = static_cast<int>(remaining_sec) % 60;
    std::ostringstream remain_oss;
    if (remain_h > 0) remain_oss << remain_h << "h";
    if (remain_m > 0) remain_oss << remain_m << "m";
    remain_oss << remain_s << "s";

    std::ostringstream output;
    output << "\r" << bar << " " << percent << "% "
           << "Saved " << filename << " "
           << "(" << frame << "/" << total_frames << ") "
           << "SPF: " << std::fixed << std::setprecision(2) << dt << "s "
           << "Remaining: " << remain_oss.str();

    std::cout << output.str() << std::flush;
}

void renderer::run_offline_rendering() {
    ShowWindow(GetHWnd(), SW_HIDE);
    sc_.set_accel_type(accel_t::uniform_grid);

    const int total_frames = save_single_frame_ ? 1 : offline_fps_ * 15;
    float angle_step = 2.0f * pi / total_frames;
    float current_angle = 0.0f;
    float total_used = 0.0f;
    auto last_time = std::chrono::steady_clock::now();

    std::cout << "Offline rendering " << total_frames << " frames to "
              << save_directory_ << std::endl;

    for (int frame = 1; frame <= total_frames; ++frame) {
        float x = trajectory_radius_ * cos(current_angle);
        float z = trajectory_radius_ * sin(current_angle);
        cam_.set_pose(
            euclidean_coordinate(x, trajectory_height_, z),
            euclidean_coordinate(0, 5, 0));
        IMAGE img = std::move(
            use_rasterize_ ? render_rasterized(false) : render(false));

        if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            if (GetAsyncKeyState('C') & 0x8000) {
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        total_used += dt;
        std::ostringstream oss;
        oss << save_directory_ << "/" << "frame_" << std::setw(3)
            << std::setfill('0') << frame << ".bmp";
        saveimage(oss.str().c_str(), &img);
        std::string time_str = format_duration(
            (total_frames - frame) * (dt + total_used / frame) * 0.5);
        update_progress(frame, total_frames, dt, oss.str());
        current_angle += angle_step;
        last_time = now;
        Sleep(1);
    }
}

void renderer::run_online_rendering() {
    if (auto_trajectory_) {
        trajectory_mode_ = true;
        trajectory_angle_ = 0.0f;
        update_trajectory_camera();
    }

    auto last_time = std::chrono::steady_clock::now();
    bool running = true;
    ExMessage msg;

    while (running) {
        while (peekmessage(&msg, EM_KEY)) {
            if (msg.message == WM_CLOSE ||
                msg.message == WM_KEYDOWN && msg.vkcode == VK_ESCAPE) {
                running = false;
                break;
            }
            handle_input(msg);
        }

        if (!IsWindow(GetHWnd())) {
            break;
        }

        if (GetAsyncKeyState(VK_CONTROL) & 0x8000) {
            if (GetAsyncKeyState('C') & 0x8000) {
                break;
            }
        }

        if (trajectory_mode_) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            if (dt > 0.1f) dt = 0.1f;
            trajectory_angle_ += 0.5f * dt;
            if (trajectory_angle_ > 2 * pi) trajectory_angle_ -= 2 * pi;
            update_trajectory_camera();
        }

        if (use_rasterize_) {
            render_rasterized(true);
        } else {
            render(true);
        }

        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        const char* accel_name = "Linear";
        if (accel_type_ == accel_t::bvh)
            accel_name = "BVH";
        else if (accel_type_ == accel_t::uniform_grid)
            accel_name = "Uniform Grid";
        if (use_rasterize_) accel_name = "Z-buffer";
        std::ostringstream oss;
        oss << "RayTracer | OBJ: " << actual_objects_
            << " | SPF: " << std::fixed << std::setprecision(2) << elapsed
            << " | Accel: " << accel_name << " | " << width_ << "x" << height_;
        std::string title = oss.str();
        SetWindowText(GetHWnd(), oss.str().c_str());
        last_time = now;
        Sleep(1);
    }
}

void renderer::handle_input(const ExMessage& msg) {
    const float move_speed = 0.2f;
    const float rot_speed = 3.0f;

    if (msg.message != WM_KEYDOWN) return;

    switch (msg.vkcode) {
        case 'W':
            cam_.move_forward(move_speed);
            break;
        case 'S':
            cam_.move_forward(-move_speed);
            break;
        case 'A':
            cam_.move_right(-move_speed);
            break;
        case 'D':
            cam_.move_right(move_speed);
            break;
        case VK_SPACE:
            cam_.move_up(move_speed);
            break;
        case VK_SHIFT:
            cam_.move_up(-move_speed);
            break;
        case 'Q':
            cam_.rotate_roll(rot_speed);
            break;
        case 'E':
            cam_.rotate_roll(-rot_speed);
            break;
        case 'I':
            cam_.rotate_pitch(rot_speed);
            break;
        case 'K':
            cam_.rotate_pitch(-rot_speed);
            break;
        case 'J':
            cam_.rotate_yaw(rot_speed);
            break;
        case 'L':
            cam_.rotate_yaw(-rot_speed);
            break;
        case 'B':
            accel_type_ = accel_t::bvh;
            sc_.set_accel_type(accel_type_);
            break;
        case 'N':
            accel_type_ = accel_t::linear;
            sc_.set_accel_type(accel_type_);
            break;
        case 'M':
            accel_type_ = accel_t::uniform_grid;
            sc_.set_accel_type(accel_type_);
            break;
        case 'R':
            build_scene();
            if (!trajectory_mode_) {
                float y_min = -2.5f;
                float y_max = (12.0f + num_objects_ / 15.0f) * 0.8f;
                float lookat_y = (y_max + y_min) * 0.5f;
                cam_.set_pose(
                    euclidean_coordinate(
                        0, trajectory_height_, trajectory_radius_),
                    euclidean_coordinate(0, lookat_y, 0));
            }
            break;
        case 'T':
            trajectory_mode_ = !trajectory_mode_;
            if (trajectory_mode_) {
                trajectory_angle_ = 0.0f;
                update_trajectory_camera();
            }
            break;
        case 'Z':
            use_rasterize_ = !use_rasterize_;
            break;
        case 'Y':
            cam_.reset_up();
            break;
        default:
            break;
    }
}

void renderer::update_trajectory_camera() {
    float x = trajectory_radius_ * cos(trajectory_angle_);
    float z = trajectory_radius_ * sin(trajectory_angle_);
    cam_.set_pose(
        euclidean_coordinate(x, trajectory_height_, z),
        euclidean_coordinate(0, 5, 0));
}

void renderer::run() {
    init_random(random_seed_);
    build_scene();

    if (!auto_trajectory_) {
        float y_min = -2.5f;
        float y_max = (12.0f + num_objects_ / 15.0f) * 0.8f;
        float lookat_y = (y_max + y_min) * 0.5f;
        cam_ = camera(
            euclidean_coordinate(0, trajectory_height_, trajectory_radius_),
            euclidean_coordinate(0, lookat_y, 0), euclidean_coordinate(0, 1, 0),
            60.0f, float(width_) / height_, 0.1f, 100.0f);
    }

    initgraph(width_, height_, EX_SHOWCONSOLE);
    setbkcolor(RGB(0, 0, 0));

    if (!save_directory_.empty()) {
        DWORD attrib = GetFileAttributes(save_directory_.c_str());
        if (attrib == INVALID_FILE_ATTRIBUTES ||
            !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
            if (!CreateDirectory(save_directory_.c_str(), NULL)) {
                std::cerr << "Cannot open or create directory " +
                                 save_directory_
                          << std::endl;
                return;
            }
        }
        run_offline_rendering();
    } else {
        run_online_rendering();
    }

    closegraph();
}

void renderer::parse_args(int argc, char** argv) {
    try {
        auto result = opt_.parse(argc, argv);
        if (result.count("help")) {
            print_help_and_exit(0);
        }
        if (result.count("preset")) {
            std::string p = result["preset"].as<std::string>();
            if (p == "low") {
                width_ = 640;
                height_ = 480;
                offline_fps_ = 6;
            } else if (p == "mid") {
                width_ = 960;
                height_ = 720;
                offline_fps_ = 12;
            } else if (p == "high") {
                width_ = 1440;
                height_ = 1080;
                offline_fps_ = 24;
            } else {
                throw cxxopts::exceptions::parsing("Invalid preset: " + p);
            }
        }
        if (result.count("resolution")) {
            std::string res = result["resolution"].as<std::string>();
            size_t xpos = res.find('x');
            if (xpos == std::string::npos) {
                throw cxxopts::exceptions::parsing(
                    "Invalid resolution format, expected WxH[:FPS]");
            }
            width_ = std::stoi(res.substr(0, xpos));
            size_t fpos = res.find(':', xpos);
            if (fpos != std::string::npos) {
                height_ = std::stoi(res.substr(xpos + 1, fpos - xpos - 1));
                offline_fps_ = std::stoi(res.substr(fpos + 1));
            } else {
                height_ = std::stoi(res.substr(xpos + 1));
            }
        }
        if (result.count("objects")) {
            num_objects_ = result["objects"].as<int>();
        }
        if (result.count("recursion")) {
            std::string dep = result["recursion"].as<std::string>();
            size_t sep = dep.find(':');
            if (sep == std::string::npos) {
                throw cxxopts::exceptions::parsing(
                    "Recursion requires MIN:MAX");
            }
            min_depth_ = std::stoi(dep.substr(0, sep));
            max_depth_ = std::stoi(dep.substr(sep + 1));
            if (min_depth_ > max_depth_) {
                throw cxxopts::exceptions::parsing(
                    "MIN_DEPTH must be <= MAX_DEPTH");
            }
        }
        if (result.count("samples")) {
            samples_per_pixel_ = result["samples"].as<int>();
        }
        if (result.count("save-dir")) {
            save_directory_ = result["save-dir"].as<std::string>();
        }
        if (result.count("save-single")) {
            save_single_frame_ = true;
            save_directory_ = ".";
        }
        if (result.count("accel")) {
            std::string acc = result["accel"].as<std::string>();
            if (acc == "linear") {
                accel_type_ = accel_t::linear;
            } else if (acc == "bvh") {
                accel_type_ = accel_t::bvh;
            } else if (acc == "uniform") {
                accel_type_ = accel_t::uniform_grid;
            } else {
                throw cxxopts::exceptions::parsing(
                    "Invalid accel type: " + acc);
            }
        }
        if (result.count("trajectory")) {
            auto_trajectory_ = true;
        }
        if (result.count("seed")) {
            random_seed_ = result["seed"].as<unsigned int>();
        }
        if (result.count("no-shadows")) {
            enable_shadows_ = false;
        } else if (result.count("shadows")) {
            enable_shadows_ = true;
        }
        if (result.count("threads")) {
            int n = result["threads"].as<int>();
            if (n < 1) n = 1;
            omp_set_num_threads(n);
        }
        if (result.count("rasterize")) {
            use_rasterize_ = true;
        }
        if (result.count("obj-type")) {
            std::string obj = result["obj-type"].as<std::string>();
            if (obj == "sphere") {
                obj_type_ = object_t::sphere;
            } else if (obj == "triangle") {
                obj_type_ = object_t::triangle;
            } else if (obj.rfind("mixed:", 0) == 0) {
                sphere_ratio_ = std::stof(obj.substr(6));
                if (sphere_ratio_ < 0.0f || sphere_ratio_ > 1.0f)
                    sphere_ratio_ = 0.5f;
                obj_type_ = object_t::mixed;
            } else {
                throw cxxopts::exceptions::parsing("Invalid obj-type: " + obj);
            }
        }
        if (result.count("mesh")) {
            mesh_filename_ = result["mesh"].as<std::string>();
            obj_type_ = object_t::mesh;
        }
        if (!result.unmatched().empty()) {
            std::cerr << "Warning: Unrecognized arguments:";
            for (const auto& u : result.unmatched()) std::cerr << " " << u;
            std::cerr << std::endl;
        }
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        print_help_and_exit(1);
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        exit(1);
    }
}

void renderer::print_help_and_exit(int code) const {
    std::cout << opt_.help() << std::endl;
    exit(code);
}
}  // namespace chrray