cmake_minimum_required(VERSION 4.0.0)
project(RT-Lab)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 23)

set(EASYX_DIR "D:/Program/easyx_26.1.1")
set(EASYX_INCLUDE ${EASYX_DIR}/include)
set(EASYX_LIB ${EASYX_DIR}/lib/VC2015/x64/*.lib)

# Add chrray-lib subdirectory
add_subdirectory(chrray-lib)

file(GLOB SOURCE_FILES "*.cpp")

add_compile_options(/utf-8)
add_executable(RT-Lab ${SOURCE_FILES})

target_link_options(RT-Lab PRIVATE 
            /SUBSYSTEM:WINDOWS 
            /ENTRY:mainCRTStartup
        )

target_include_directories(RT-Lab PRIVATE ${EASYX_INCLUDE})
target_link_libraries(RT-Lab ${EASYX_LIB} chrray-lib)