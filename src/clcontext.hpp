#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#define RGB(r,g,b) float4(r / 255.0f, g / 255.0f, b / 255.0f, 0.0f)

#include "cl2.hpp" // first due to conflicts with X.h (included by glxew)

#if defined(__APPLE__)
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/OpenGL.h>
#elif defined(__linux__)
#include <GL/glxew.h>
#endif

#include <GLFW/glfw3.h> // texture conversion stuff
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "math/float3.hpp"
#include "kernelreader.hpp"
#include "geom.h"

using FireRays::float4;

// Test scene
static Sphere test_spheres[] =
{
    // radius, position, Kd
    { 1.0f, float4(0.0f, 0.0f, 0.0f, 0.0f), RGB(255, 0, 0) },             // big sphere
    { 0.5f, float4(0.0f, 1.5f, 0.0f, 0.0f), RGB(0, 255, 0) },             // small sphere
    { 1000.0f, float4(0.0f, -1000.0f, 0.0f, 0.0f), RGB(0, 0, 255) },      // floor
    { 1000.0f, float4(0.0f, 0.0f, -1005.0f, 0.0f), RGB(180, 190, 180) },  // back wall
    { 1000.0f, float4(-1005.0f, 0.0f, 0.0f, 0.0f), RGB(205, 110, 15) },   // left wall
    { 1000.0f, float4(+1005.0f, 0.0f, 0.0f, 0.0f), RGB(255, 0, 255) },    // right wall
};

class CLContext
{
public:
    CLContext(GLuint gl_PBO);
    ~CLContext();

    void executeKernel(const RenderParams &params);
    void setupParams();
    void updateParams(const RenderParams &params);
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
    cl::Buffer renderParams;            // contains only one RenderParam
};