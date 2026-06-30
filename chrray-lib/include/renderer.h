#include <camera.h>
#define NOMINMAX
#include <graphics.h>
#include <light.h>
#include <object.h>
#include <omp.h>
#include <scene.h>
#include <texture.h>
#include <utils.h>
#include <cxxopts.hpp>

namespace chrray {
class renderer {
public:
    renderer();
    void parse_args(int argc, char** argv);
    void run();

private:
    void print_help_and_exit(int code) const;
    void error_and_exit(const std::string& msg) const;

    void build_scene();
    color ray_color(const ray& r, int depth) const;
    IMAGE render(bool draw);
    IMAGE render_rasterized(bool draw);
    void run_offline_rendering();
    void run_online_rendering();
    void handle_input(const ExMessage& msg);
    void update_trajectory_camera();

private:
    int width_ = 960;
    int height_ = 720;
    int min_depth_ = 4;
    int max_depth_ = 8;
    int samples_per_pixel_ = 4;

    accel_t accel_type_ = accel_t::bvh;
    bool use_rasterize_ = false;
    bool enable_shadows_ = true;
    unsigned int random_seed_ = 0;

    int num_objects_ = 100;
    enum class object_t {
        sphere,
        triangle,
        mixed,
        mesh,
    } obj_type_ = object_t::sphere;
    float sphere_ratio_ = 0.5f;
    int actual_objects_ = 0;
    std::string mesh_filename_;

    bool auto_trajectory_ = false;
    float trajectory_angle_ = 0.0f;
    float trajectory_radius_ = 25.0f;
    float trajectory_height_ = 8.0f;
    bool trajectory_mode_ = false;

    int offline_fps_ = 24;
    bool save_single_frame_ = false;
    std::string save_directory_;
    int priority_ = 1;

    cxxopts::Options opt_;
    scene sc_;
    camera cam_;
};

}  // namespace chrray