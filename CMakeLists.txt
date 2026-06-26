cmake_minimum_required(VERSION 4.0.0)
project(RT-Lab)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 23)

set(EASYX_DIR "D:/Program/easyx_26.1.1")
set(EASYX_INCLUDE ${EASYX_DIR}/include)
set(EASYX_LIB ${EASYX_DIR}/lib/VC2015/x64/EasyXa.lib)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.3.1                       
)

FetchContent_MakeAvailable(cxxopts)

add_compile_options(/utf-8 /fp:fast /arch:AVX2)
add_subdirectory(chrray-lib)

file(GLOB SOURCE_FILES "*.cpp")

add_executable(RT-Lab ${SOURCE_FILES})

target_include_directories(RT-Lab PRIVATE ${EASYX_INCLUDE})
target_link_libraries(RT-Lab PRIVATE 
    ${EASYX_LIB} 
    chrray-lib 
    OpenMP::OpenMP_CXX
)