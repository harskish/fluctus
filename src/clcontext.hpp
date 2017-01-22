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
#include "settings.hpp"
#include "bvh.hpp"
#include "scene.hpp"

using FireRays::float3;

// Test scene
static Sphere test_spheres[] =
{
    // radius, position, Kd
    //{ 1.0f, float4(0.0f, 0.0f, 0.0f, 0.0f), RGB2f3(255, 0, 0) },             // big sphere
    //{ 0.0f, float4(0.0f, 1.5f, 0.0f, 0.0f), RGB2f3(255, 255, 255) },             // small sphere
    { 1000.0f, float4(0.0f, 0.0f, +1008.0f, 0.0f), RGB2f3(180, 190, 180) },  // back wall
    { 1000.0f, float4(0.0f, 0.0f, -1008.0f, 0.0f), RGB2f3(180, 190, 180) },  // front (visible) wall
    { 1000.0f, float4(0.0f, -1001.0f, 0.0f, 0.0f), RGB2f3(0, 0, 255) },      // floor
    { 1000.0f, float4(-1008.0f, 0.0f, 0.0f, 0.0f), RGB2f3(205, 110, 15) },   // left wall
    { 1000.0f, float4(+1008.0f, 0.0f, 0.0f, 0.0f), RGB2f3(255, 0, 255) },    // right wall
    { 1000.0f, float4(0.0f, +1020.0f, 0.0f, 0.0f), RGB2f3(0, 0, 255) },      // ceiling
};

static PointLight test_lights[] =
{
    { RGB2f3(255, 255, 255) * 30.0f, {float3(2.0f, 4.0f, 0.0f)} },
    //{ RGB2f3(255, 255, 255) * 30.0f, { float3(0.0f, 10.0f, 0.0f) } },
};

class CLContext
{

friend class Tracer;

public:
    CLContext(GLuint *textures);
    ~CLContext();

    void executeMegaKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration);
    void executeRayGenKernel(const RenderParams &params);
    void executeNextVertexKernel(const RenderParams &params);
    void executeSplatKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration);

    void setupParams();
    void updateParams(const RenderParams &params);
    void uploadSceneData(BVH *bvh, Scene *scene);
    void createTextures(GLuint *tex_arr);
    void createEnvMap(float *data, int width, int height);
private:
    void printDevices();
    void setupScene();
    void verify(std::string msg);
    void packTextures(Scene *scene);
    
    void setupKernels();
    void setupRayGenKernel();
    void setupNextVertexKernel();
    void setupSplatKernel();
    void setupMegaKernel();
    void initMCBuffers();

    void buildKernel(cl::Kernel &target, std::string fileName, std::string methodName);

    cl::Platform &getPlatformByName(std::vector<cl::Platform> &platforms, std::string name);
    cl::Device &getDeviceByName(std::vector<cl::Device> &devices, std::string name);
    std::string errorString();

    int err;                                // error code returned from api calls
    size_t ndRangeSizes[2];                 // kernel workgroup sizes
    const size_t NUM_TASKS = 1920 * 1080;   // the amount of paths in flight simultaneously, limited by VRAM

    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    
    // Kernels
    cl::Kernel kernel_monolith;
    cl::Kernel mk_raygen;
    cl::Kernel mk_next_vertex;
    cl::Kernel mk_splat;

    std::vector<cl::Memory> sharedMemory;   // device memory used for pixel data (two textures)
    cl::Buffer sphereBuffer;
    cl::Buffer lightBuffer;
    cl::Buffer renderParams;                // contains only one RenderParam
    
    // Microkernel buffers
    cl::Buffer raysBuffer;
    cl::Buffer tasksBuffer;

    cl::Image2D environmentMap;

    // Variables from BVH
    cl::Buffer triangleBuffer;
    cl::Buffer nodeBuffer;
    cl::Buffer indexBuffer;
    cl::Buffer materialBuffer;
    cl::Buffer texDescriptorBuffer;
    cl::Buffer texDataBuffer;
    cl_uint nodes = 0; // needed outside of building?
};