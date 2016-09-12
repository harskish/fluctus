#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#define RGB2f3(r,g,b) float4(r / 255.0f, g / 255.0f, b / 255.0f)

#include "cl2.hpp" // first due to conflicts with X.h (included by glxew)

#if defined(__APPLE__)
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/OpenGL.h>
#elif defined(__linux__)
#include <GL/glxew.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <GL/glew.h> //needed?
#include <GL/wglew.h>
#endif

#include <GLFW/glfw3.h> // texture conversion stuff
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "math/float3.hpp"
#include "kernelreader.hpp"
#include "geom.h"
#include "triangle.hpp"
#include "bvhnode.hpp"

using FireRays::float3;

// Test scene
static Sphere test_spheres[] =
{
    // radius, position, Kd
    //{ 1.0f, float4(0.0f, 0.0f, 0.0f, 0.0f), RGB2f4(255, 0, 0) },             // big sphere
    { 0.0f, float4(0.0f, 1.5f, 0.0f, 0.0f), RGB2f3(0, 255, 0) },             // small sphere
    //{ 1000.0f, float4(0.0f, -1005.0f, 0.0f, 0.0f), RGB2f4(0, 0, 255) },      // floor
    //{ 1000.0f, float4(0.0f, 0.0f, -1008.0f, 0.0f), RGB2f4(180, 190, 180) },  // front (visible) wall
    //{ 1000.0f, float4(-1008.0f, 0.0f, 0.0f, 0.0f), RGB2f4(205, 110, 15) },   // left wall
    //{ 1000.0f, float4(+1008.0f, 0.0f, 0.0f, 0.0f), RGB2f4(255, 0, 255) },    // right wall
    //{ 1000.0f, float4(0.0f, 0.0f, +1008.0f, 0.0f), RGB2f4(180, 190, 180) },  // back wall
    //{ 1000.0f, float4(0.0f, +1020.0f, 0.0f, 0.0f), RGB2f4(0, 0, 255) },      // ceiling
};

static Light test_lights[] =
{
    { RGB2f3(255, 255, 255) * 40.0f, {float3(0.0f, 16.5f, 0.0f)}, {float3(0.0f)}, L_POINT },
    { RGB2f3(255, 255, 255) * 30.0f, {float3(2.0f, 4.0f, 0.0f)}, {float3(0.0f)}, L_POINT },
};

class CLContext
{

friend class Tracer;

public:
    CLContext(GLuint gl_PBO);
    ~CLContext();

    void executeKernel(const RenderParams &params);
    void setupParams();
    void updateParams(const RenderParams &params);
    void createBVHBuffers(std::vector<RTTriangle> *triangles, std::vector<cl_uint> *indices, std::vector<Node> *nodes);
    void createPBO(GLuint gl_PBO);
    void createEnvMap(float *data, int width, int height);
private:
    void printDevices();
    void setupScene();
    void verify(std::string msg);
    std::string errorString();

    int err;                                // error code returned from api calls
    size_t ndRangeSizes[2];                 // kernel workgroup sizes

    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    cl::Kernel pt_kernel;

    std::vector<cl::Memory> sharedMemory;   // device memory used for pixel data
    cl::Buffer sphereBuffer;
    cl::Buffer lightBuffer;
    cl::Buffer renderParams;                // contains only one RenderParam

    cl::Image2D environmentMap;

    // Variables from BVH
    cl::Buffer triangleBuffer;
    cl::Buffer nodeBuffer;
    cl::Buffer indexBuffer;
    cl_uint nodes = 0; // needed outside of building?
};