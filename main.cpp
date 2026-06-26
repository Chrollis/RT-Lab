#include <omp.h>
#include <renderer.h>
#include <iostream>

int main(int argc, char** argv) {
    omp_set_num_threads(omp_get_max_threads());

    chrray::renderer app;
    app.parse_args(argc, argv);
    app.run();

    return 0;
}