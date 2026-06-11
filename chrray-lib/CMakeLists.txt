cmake_minimum_required(VERSION 4.0.0)
project(chrray-lib)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 23)

file(GLOB SOURCE_FILES "src/*.cpp")

add_library(chrray-lib ${SOURCE_FILES})

target_include_directories(chrray-lib PUBLIC include)