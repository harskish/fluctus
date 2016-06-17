#pragma once

#define DATA_SIZE (2048)
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/OpenGL.h>
#endif

#include <GLFW/glfw3.h> // texture conversion stuff
#include <iostream>
#include <string>
#include "cl2.hpp"
#include "kernelreader.hpp"
#include "geom.h"

// Test scene
static Sphere test_spheres[2] =
{
    // radius, position, Kd
    { 1.0f, {{ 0.0f, 0.0f, 0.0f, 0.0f }}, {{ 1.0f, 0.0f, 0.0f, 0.0f }} },
    { 0.5f, {{ 0.0f, 1.5f, 0.0f, 0.0f }}, {{ 0.0f, 1.0f, 0.0f, 0.0f }} }
};

class CLContext
{
public:
    CLContext(GLuint gl_PBO);
    ~CLContext();

    void executeKernel(const unsigned int width, const unsigned int height);
    void createPBO(GLuint gl_PBO);
private:
    void printDevices();
    void setupScene();
    std::string errorString();

    int err;                            // error code returned from api calls
    size_t ndRangeSizes[2];             // kernel workgroup sizes

    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    cl::Kernel pt_kernel;
    
    cl_mem cl_PBO = 0;                  // device memory used for pixel data
    cl::Buffer sphereBuffer;
};