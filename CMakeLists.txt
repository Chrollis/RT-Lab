cmake_minimum_required(VERSION 3.10)
project(RT-Lab)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 23)

set(EASYX_DIR "D:/Program/easyx_26.1.1")
set(EASYX_INCLUDE ${EASYX_DIR}/include)
set(EASYX_LIB ${EASYX_DIR}/lib/VC2015/x64/*.lib)

file(GLOB_RECURSE SOURCE_FILES "*.cpp" "chrray-lib/src/*.cpp")

add_compile_options(/utf-8)
add_executable(RT-Lab ${SOURCE_FILES})

target_link_options(RT-Lab PRIVATE 
            /SUBSYSTEM:WINDOWS 
            /ENTRY:mainCRTStartup
        )

target_include_directories(RT-Lab 
        PRIVATE 
        ${EASYX_INCLUDE}
        ./chrray-lib/include
    )
target_link_libraries(RT-Lab ${EASYX_LIB})