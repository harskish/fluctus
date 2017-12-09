#pragma once

#ifdef _DEBUG
#define CPU_DEBUGGING
#endif

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
#include <glad/glad.h>
#include <Windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h> // texture conversion stuff
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include "geom.h"
#include "math/float3.hpp"
#include "kernelreader.hpp"
#include "geom.h"
#include "triangle.hpp"
#include "bvhnode.hpp"
#include "settings.hpp"
#include "bvh.hpp"
#include "scene.hpp"
#include "texture.hpp"
#include "window.hpp"
#include "utils.h"

using FireRays::float3;

// Test scene
static Sphere test_spheres[] =
{
    // radius, position, Kd
    //{ 1.0f, float4(0.0f, 0.0f, 0.0f, 0.0f), RGB2f3(255, 0, 0) },             // big sphere
    //{ 0.0f, float4(0.0f, 1.5f, 0.0f, 0.0f), RGB2f3(255, 255, 255) },             // small sphere
    { float4(0.0f, 0.0f, +1008.0f, 0.0f), RGB2f3(180, 190, 180), 1000.0f },  // back wall
    { float4(0.0f, 0.0f, -1008.0f, 0.0f), RGB2f3(180, 190, 180), 1000.0f },  // front (visible) wall
    { float4(0.0f, -1001.0f, 0.0f, 0.0f), RGB2f3(0, 0, 255), 1000.0f },      // floor
    { float4(-1008.0f, 0.0f, 0.0f, 0.0f), RGB2f3(205, 110, 15), 1000.0f },   // left wall
    { float4(+1008.0f, 0.0f, 0.0f, 0.0f), RGB2f3(255, 0, 255), 1000.0f },    // right wall
    { float4(0.0f, +1020.0f, 0.0f, 0.0f), RGB2f3(0, 0, 255), 1000.0f },      // ceiling
};

static PointLight test_lights[] =
{
    { RGB2f3(255, 255, 255) * 30.0f, {float3(2.0f, 4.0f, 0.0f)} },
    //{ RGB2f3(255, 255, 255) * 30.0f, { float3(0.0f, 10.0f, 0.0f) } },
};

class PTWindow;
class CLContext
{

friend class Tracer;

public:
    CLContext();
    ~CLContext();

    void enqueueMegaKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration);
	void enqueueResetKernel(const RenderParams &params);
	void enqueueRayGenKernel(const RenderParams &params);
    void enqueueNextVertexKernel(const RenderParams &params);
    void enqueueExplSampleKernel(const RenderParams &params, const cl_uint iteration);
    void enqueueSplatKernel(const RenderParams &params, const cl_uint iteration);
    void enqueueSplatPreviewKernel(const RenderParams &params);
    void finishQueue();

    void setup(PTWindow *window);
    void setupParams();
    void setupStats();
    void resetStats();
    void fetchStatsAsync();
    const RenderStats getStats();

    void updateParams(const RenderParams &params);
    void uploadSceneData(BVH *bvh, Scene *scene);
    void setupPixelStorage(GLuint *tex_arr, GLuint gl_PBO);
	void saveImage(std::string filename, const RenderParams &params, bool usingMicroKernel);
    void createEnvMap(EnvironmentMap *map);
private:
    void printDevices();
    void setupScene();
    void verify(std::string msg, int pred = -1);
    void packTextures(Scene *scene);
    
    void setupKernels();
	void setupResetKernel();
    void setupRayGenKernel();
    void setupNextVertexKernel();
    void setupExplSampleKernel();
    void setupSplatKernel();
    void setupSplatPreviewKernel();
    void setupMegaKernel();
    void initMCBuffers();

    void buildKernel(cl::Kernel &target, std::string fileName, std::string methodName);

    cl::Platform &getPlatformByName(std::vector<cl::Platform> &platforms, std::string name);
    cl::Device &getDeviceByName(std::vector<cl::Device> &devices, std::string name);
    std::string errorString();

    int err;                                // error code returned from api calls
    size_t ndRangeSizes[2];                 // kernel workgroup sizes
#ifdef CPU_DEBUGGING
    const cl_uint NUM_TASKS = 1;
#else
    const cl_uint NUM_TASKS = 2 << 19;   // the amount of paths in flight simultaneously, limited by VRAM
#endif

    // For showing progress
    PTWindow *window;
    
    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    
    // Kernels
    cl::Kernel kernel_monolith;
	cl::Kernel mk_reset;
    cl::Kernel mk_raygen;
    cl::Kernel mk_next_vertex;
    cl::Kernel mk_sample_explicit;
    cl::Kernel mk_splat;
    cl::Kernel mk_splat_preview;

    // Pixel storage
    cl::BufferGL pixelBuffer;
    cl::ImageGL frontBuffer;
    cl::ImageGL backBuffer;
    std::vector<cl::Memory> sharedMemory;   // device memory used for pixel data (tex1, tex2, mk_pixelbuffer)
    
    cl::Buffer sphereBuffer;
    cl::Buffer lightBuffer;
    cl::Buffer renderParams;                // contains only one RenderParam
    cl::Buffer renderStats;                 // ray + sample counts
    RenderStats statsAsync;                 // fetched asynchronously from device after each iteration
    
    // Microkernel buffer
    cl::Buffer tasksBuffer;

	// Environment map data
    cl::Image2D environmentMap;
	cl::Buffer probTable;
	cl::Buffer aliasTable;
	cl::Buffer pdfTable;

    // Variables from BVH
    cl::Buffer triangleBuffer;
    cl::Buffer nodeBuffer;
    cl::Buffer indexBuffer;
    cl::Buffer materialBuffer;
    cl::Buffer texDescriptorBuffer;
    cl::Buffer texDataBuffer;
    cl_uint nodes = 0; // needed outside of building?
};