cmake_minimum_required(VERSION 3.16)
project(deformation)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Default build type: Release" FORCE)
endif()
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_WITH_TETGEN            "Use TetGen"         ON)
include(libigl)

# Add your project files
add_executable(bc bc.cpp)
add_executable(bc_interp bc_interp.cpp)
add_executable(sample_ctrl_pts sample_key_points.cpp)
target_link_libraries(bc PUBLIC igl::core igl::tetgen)
target_link_libraries(bc_interp PUBLIC igl::core igl::tetgen)
target_link_libraries(sample_ctrl_pts PUBLIC igl::core)

