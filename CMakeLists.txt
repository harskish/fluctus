cmake_minimum_required(VERSION 3.3)
project(CL_TRACE)

set(CMAKE_LIBRARY_PATH /usr/local/cuda-8.0/lib64/)
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

include_directories(
    ${GLEW_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    include
)

set(SOURCE_FILES
    src/main.cpp
    src/clcontext.cpp
    src/clcontext.hpp
    src/window.cpp
    src/window.hpp
    src/geom.h
    src/kernel.cl
    src/kernelreader.cpp
    src/kernelreader.hpp
    src/tracer.cpp
    src/tracer.hpp src/bvh.hpp src/bvh.cpp src/bvhnode.hpp src/bvhnode.cpp src/rtutil.hpp src/rttriangle.hpp src/raycastresult.hpp include/math/int3.hpp)

add_executable(CL_TRACE ${SOURCE_FILES})

target_link_libraries(CL_TRACE
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARY}
    ${OpenCL_LIBRARY}
    ${GLFW_STATIC_LIBRARIES}
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib")
endif()