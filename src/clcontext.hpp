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

static PointLight test_lights[] =
{
    { RGB2f3(255, 255, 255) * 30.0f, {float3(2.0f, 4.0f, 0.0f)} },
    //{ RGB2f3(255, 255, 255) * 30.0f, { float3(0.0f, 10.0f, 0.0f) } },
};

typedef struct
{
    float primary = 0.0f;
    float extension = 0.0f;
    float shadow = 0.0f;
    float samples = 0.0f;
    float total = 0.0f;
} PerfNumbers;

class PTWindow;
class CLContext
{

friend class Tracer;

public:
    CLContext();
    ~CLContext();

	void enqueueResetKernel(const RenderParams &params);
	void enqueueRayGenKernel(const RenderParams &params);
    void enqueueNextVertexKernel(const RenderParams &params);
    void enqueueBsdfSampleKernel(const RenderParams &params, const cl_uint iteration);
    void enqueueSplatKernel(const RenderParams &params, const cl_uint iteration);
    void enqueueSplatPreviewKernel(const RenderParams &params);
    void enqueuePostprocessKernel(const RenderParams &params);
    
    void enqueueWfResetKernel(const RenderParams &params);
    void enqueueWfRaygenKernel(const RenderParams &params);
    void enqueueWfExtRayKernel(const RenderParams &params);
    void enqueueWfShadowRayKernel(const RenderParams &params);
    void enqueueWfLogicKernel(const bool firstIteration);
    void enqueueWfMaterialKernels(const RenderParams &params);
   
    void enqueueClearWfQueues();
    void finishQueue();
    void updatePixelIndex(cl_uint numPixels, cl_uint numNewPaths);
    void resetPixelIndex();

    Hit pickSingle(float NDCx, float NDCy);

    void setup(PTWindow *window);
    void setupParams();
    void setupPickResult();
    void setupStats();
    void resetStats();
    void fetchStatsAsync();
    void updateRenderPerf(float deltaT);
    const PerfNumbers getRenderPerf();
    const RenderStats getStats();
    void enqueueGetCounters(QueueCounters *cnt);

    void checkTracingPerf();

    void updateParams(const RenderParams &params);
    void uploadSceneData(BVH *bvh, Scene *scene);
    void setupPixelStorage(PTWindow *window);
	void saveImage(std::string filename, const RenderParams &params);
    void createEnvMap(EnvironmentMap *map);
private:
    void printDevices();
    void setupScene();
    void verify(std::string msg, int pred = -1);
    void packTextures(Scene *scene);

    void enqueueWfDiffuseKernel(const RenderParams &params);
    void enqueueWfGlossyKernel(const RenderParams &params);
    void enqueueWfGGXReflKernel(const RenderParams &params);
    void enqueueWfGGXRefrKernel(const RenderParams &params);
    void enqueueWfDeltaKernel(const RenderParams &params);
    
    void setupKernels();
	void setupResetKernel();
    void setupRayGenKernel();
    void setupNextVertexKernel();
    void setupBsdfSampleKernel();
    void setupSplatKernel();
    void setupSplatPreviewKernel();
    void setupPostprocessKernel();
    void setupPickKernel();
    void setupWfExtKernel();
    void setupWfResetKernel();
    void setupWfLogicKernel();
    void setupWfShadowKernel();
    void setupWfRaygenKernel();
    void setupWfDiffuseKernel();
    void setupWfGlossyKernel();
    void setupWfGGXReflKernel();
    void setupWfGGXRefrKernel();
    void setupWfDeltaKernel();
    void initMCBuffers();

    void buildKernel(cl::Kernel &target, std::string fileName, std::string methodName);

    cl::Platform &getPlatformByName(std::vector<cl::Platform> &platforms, std::string name);
    cl::Device &getDeviceByName(std::vector<cl::Device> &devices, std::string name);
    std::string errorString();

    int err;                                // error code returned from api calls
    size_t ndRangeSizes[2];                 // kernel workgroup sizes
    
    cl_uint NUM_TASKS = 0;  // the amount of paths in flight simultaneously, limited by VRAM, defined in settings

    // For showing progress
    PTWindow *window;
    
    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    
    // General kernels
    cl::Kernel kernel_pick;
    cl::Kernel mk_postprocess;

    // Luxrender-style microkernels
    cl::Kernel mk_reset;
    cl::Kernel mk_raygen;
    cl::Kernel mk_next_vertex;
    cl::Kernel mk_sample_bsdf;
    cl::Kernel mk_splat;
    cl::Kernel mk_splat_preview;
    
    // Aila-style wavefront kernels
    cl::Kernel wf_reset;
    cl::Kernel wf_extension;
    cl::Kernel wf_raygen;
    cl::Kernel wf_logic;
    cl::Kernel wf_shadow;
    cl::Kernel wf_diffuse;
    cl::Kernel wf_glossy;
    cl::Kernel wf_ggx_refl;
    cl::Kernel wf_ggx_refr;
    cl::Kernel wf_delta;

    // Pixel storage
    cl::Buffer pixelBuffer;     // raw (linear) pixel data, not used by OpenGL
    cl::Buffer denoiserAlbedoBuffer;
    cl::Buffer denoiserNormalBuffer;
    cl::BufferGL previewBuffer; // post-processed buffer, shown on screen
    cl::BufferGL denoiserAlbedoBufferGL;
    cl::BufferGL denoiserNormalBufferGL;
    std::vector<cl::Memory> sharedMemory;   // device memory used for pixel data (previewBuffer)
    
    cl::Buffer lightBuffer;
    cl::Buffer pickResult;
    cl::Buffer renderParams;                // contains only one RenderParam
    
    // Performance statistics
    cl::Buffer renderStats;                 // ray + sample counts
    RenderStats statsAsync;                 // fetched asynchronously from device after each iteration
    PerfNumbers renderPerf;
    cl::Event extRayEvent;
    cl::Event shdwRayEvent;
    
    // Microkernel buffers
    cl::Buffer tasksBuffer;
    cl::Buffer raygenQueue;     // indices of paths to regenerate
    cl::Buffer extensionQueue;  // indices of paths to extend
    cl::Buffer shadowQueue;     // indices of shadow ray casts
    cl::Buffer diffuseMatQueue;
    cl::Buffer glossyMatQueue;
    cl::Buffer ggxReflMatQueue;
    cl::Buffer ggxRefrMatQueue;
    cl::Buffer deltaMatQueue;
    cl::Buffer currentPixelIdx; // points to next pixel, since NUM_TASKS != #pixels
    cl::Buffer queueCounters;   // atomic counters keeping track of queue lengths
    QueueCounters hostCounters = {};
    cl_uint pixelIdx = 0;

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
