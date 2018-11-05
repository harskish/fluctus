#pragma once

#ifdef _DEBUG
#define CPU_DEBUGGING
#endif

#include "cl2.hpp"
#include "geom.h"
#include "Kernel.hpp"
#include <string>

typedef struct
{
    float primary = 0.0f;
    float extension = 0.0f;
    float shadow = 0.0f;
    float samples = 0.0f;
    float total = 0.0f;
} PerfNumbers;

class EnvironmentMap;
class BVH;
class Scene;
class PTWindow;

class CLContext
{

friend class Tracer;

public:
    CLContext();
    ~CLContext() = default;

	void enqueueResetKernel(const RenderParams &params);
	void enqueueRayGenKernel(const RenderParams &params);
    void enqueueNextVertexKernel(const RenderParams &params);
    void enqueueBsdfSampleKernel(const RenderParams &params);
    void enqueueSplatKernel(const RenderParams &params);
    void enqueueSplatPreviewKernel(const RenderParams &params);
    void enqueuePostprocessKernel(const RenderParams &params);
    
    void enqueueWfResetKernel(const RenderParams &params);
    void enqueueWfRaygenKernel(const RenderParams &params);
    void enqueueWfExtRayKernel(const RenderParams &params);
    void enqueueWfShadowRayKernel(const RenderParams &params);
    void enqueueWfLogicKernel(const RenderParams &params, const bool firstIteration);
    void enqueueWfMaterialKernels(const RenderParams &params);

    // Done conservatively
    void recompileKernels(bool setArgs);
   
    void enqueueClearWfQueues();
    void finishQueue();
    void updatePixelIndex(cl_uint numPixels, cl_uint numNewPaths);
    void resetPixelIndex();
    cl_uint getNumTasks() const;

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
    void enqueueWfAllMaterialsKernel(const RenderParams &params);
    
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
    void setupWfAllMaterialsKernel();
    void initMCBuffers();

    void setKernelBuildSettings();

    cl::Platform &getPlatformByName(std::vector<cl::Platform> &platforms, std::string name);
    cl::Device &getDeviceByName(std::vector<cl::Device> &devices, std::string name);

    int err;                // error code returned from api calls
    cl_uint NUM_TASKS = 0;  // the amount of paths in flight simultaneously, limited by VRAM, defined in settings

    // For showing progress
    PTWindow *window;
    
    std::vector<cl::Device> clDevices;
    cl::Device device;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    
    // General kernels
    flt::Kernel* kernel_pick = nullptr;
    flt::Kernel* mk_postprocess = nullptr;

    // Luxrender-style microkernels
    flt::Kernel* mk_reset = nullptr;
    flt::Kernel* mk_raygen = nullptr;
    flt::Kernel* mk_next_vertex = nullptr;
    flt::Kernel* mk_sample_bsdf = nullptr;
    flt::Kernel* mk_splat = nullptr;
    flt::Kernel* mk_splat_preview = nullptr;
    
    // Aila-style wavefront kernels
    flt::Kernel* wf_reset = nullptr;
    flt::Kernel* wf_extension = nullptr;
    flt::Kernel* wf_raygen = nullptr;
    flt::Kernel* wf_logic = nullptr;
    flt::Kernel* wf_shadow = nullptr;
    flt::Kernel* wf_diffuse = nullptr;
    flt::Kernel* wf_glossy = nullptr;
    flt::Kernel* wf_ggx_refl = nullptr;
    flt::Kernel* wf_ggx_refr = nullptr;
    flt::Kernel* wf_delta = nullptr;
    flt::Kernel* wf_mat_all = nullptr;

    
    // Device memory shared with GL
    std::vector<cl::Memory> sharedMemory;
    
    // Performance statistics
    RenderStats statsAsync;  // fetched asynchronously from device after each iteration
    PerfNumbers renderPerf;
    cl::Event extRayEvent;
    cl::Event shdwRayEvent;
    QueueCounters hostCounters = {}; // synced from queueCounters
    cl_uint pixelIdx = 0;

public:

    // Device buffers need to be accessible to kernel implementations
    struct
    {
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

        // Variables from BVH
        cl::Buffer triangleBuffer;
        cl::Buffer nodeBuffer;
        cl::Buffer indexBuffer;
        cl::Buffer materialBuffer;
        cl::Buffer texDescriptorBuffer;
        cl::Buffer texDataBuffer;

        // Environment map data
        cl::Image2D environmentMap;
        cl::Buffer probTable;
        cl::Buffer aliasTable;
        cl::Buffer pdfTable;

        // Statistics
        cl::Buffer renderStats;  // ray + sample counts

        // Pixel storage
        cl::Buffer pixelBuffer;     // raw (linear) pixel data, not used by OpenGL
        cl::Buffer denoiserAlbedoBuffer;
        cl::Buffer denoiserNormalBuffer;
        cl::BufferGL previewBuffer; // post-processed buffer, shown on screen
        cl::BufferGL denoiserAlbedoBufferGL;
        cl::BufferGL denoiserNormalBufferGL;

        // Single element buffers
        cl::Buffer pickResult;
        cl::Buffer renderParams;
    } deviceBuffers;
};
