cmake_minimum_required(VERSION 3.3)
project(CL_TRACE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib -lglfw3 -framework OpenCL -framework OpenGL")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

include_directories(include /usr/local/include)

set(SOURCE_FILES
    include/cl2.hpp
    src/clcontext.cpp
    src/clcontext.hpp
    src/window.cpp
    src/window.hpp
    src/geom.h
    src/kernel.cl
    src/kernelreader.cpp
    src/kernelreader.hpp
    src/main.cpp)

add_executable(CL_TRACE ${SOURCE_FILES})