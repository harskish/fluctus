#include "clcontext.hpp"

inline bool platformIsNvidia(cl::Platform platform)
{
    std::string name = platform.getInfo<CL_PLATFORM_NAME>();
    return name.find("NVIDIA") != std::string::npos;
}

inline std::string getAbsolutePath(std::string filename)
{
    const int MAX_LENTH = 4096;
    char resolved_path[MAX_LENTH];
    #ifdef _WIN32
        _fullpath(resolved_path, filename.c_str(), MAX_LENTH);
    #else
        realpath(filename.c_str(), resolved_path);
    #endif
    return std::string(resolved_path);
}

CLContext::CLContext()
{
    printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    platform = getPlatformByName(platforms, Settings::getInstance().getPlatformName());
    std::cout << "PLATFORM: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	#ifdef CPU_DEBUGGING
		platform.getDevices(CL_DEVICE_TYPE_CPU, &clDevices);
	#else
		platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);
	#endif

    // Init shared context
    #ifdef __APPLE__
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)kCGLShareGroup, 0
        };
    #else
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
        #if defined(__linux__)
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        #elif defined(_WIN32)
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        #endif
            0
        };
    #endif

    // Select correct device from context based on settings
    device = getDeviceByName(clDevices, Settings::getInstance().getDeviceName());
    std::cout << "DEVICE: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // Restrict context to selected device
    clDevices = { device };
    context = cl::Context(clDevices, props, NULL, NULL, &err);
    verify("Failed to create shared context");

    // Create command queue for context
    cmdQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    verify("Failed to create command queue!");

    // Setup WF task buffer size
    cl_uint bufferSize = Settings::getInstance().getWfBufferSize();
    NUM_TASKS = bufferSize;
}

void CLContext::setup(PTWindow *window)
{
    this->window = window;

    // Setup RenderParams
    setupParams();

    // Setup pick result buffer
    setupPickResult();

    // Setup RenerStats
    setupStats();

    // Create OpenCL buffer from OpenGL PBO
    setupPixelStorage(window);

    // Allocate device memory for scene
    setupScene();

    // Build kernels, set their params
    initMCBuffers();
    setupKernels();
}

void CLContext::setupKernels()
{
    // Microkernels
	setupResetKernel();
    setupRayGenKernel();
    setupNextVertexKernel();
    setupBsdfSampleKernel();
    setupSplatKernel();
    setupSplatPreviewKernel();

    // Wavefront kernels
    setupWfResetKernel();
    setupWfExtKernel();
    setupWfRaygenKernel();
    setupWfLogicKernel();
    setupWfShadowKernel();
    setupWfDiffuseKernel();
    setupWfGlossyKernel();
    setupWfGGXReflKernel();
    setupWfGGXRefrKernel();
    setupWfDeltaKernel();

    // Other
    setupPickKernel();
    setupPostprocessKernel();
}

// For copying SoA data to host
inline void copyToHost(GPUTaskState *dst, GPUTaskState *src, size_t NUM_TASKS)
{
    float *hostData = (float*)src;

    for (int i = 0; i < NUM_TASKS; i++)
    {
        GPUTaskState curr;
        for (int j = 0; j < sizeof(GPUTaskState) / sizeof(float); j++)
        {
            ((float*)&curr)[j] = hostData[j * NUM_TASKS + i];
        }
        dst[i] = curr;
    }
}

// Init state buffers (rays, tasks) needed by microkernels
void CLContext::initMCBuffers()
{
    // TODO: ensure 32bit divisibility in SoA mode
    const size_t t_bytes = NUM_TASKS * sizeof(GPUTaskState);
    tasksBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, t_bytes, NULL, &err);
    verify("Task buffer creation failed!");

    // Queues
    cl_uint pixelIndex = 0;

    // TODO: CL_MEM_USE_HOST_PTR for seeing queues on host
    currentPixelIdx = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(cl_uint), (void*)&pixelIndex, &err);
    queueCounters = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(QueueCounters), (void*)&hostCounters, &err);
    raygenQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    extensionQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    shadowQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    diffuseMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    glossyMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    ggxReflMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    ggxRefrMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deltaMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    verify("MK queue creation failed");

    const size_t memoryUsageMiB = t_bytes / (2 << 19);
    std::cout << "Microkernel state data: " << memoryUsageMiB << " MiB" << std::endl;
}

// Build kernel based on file name and entrypoint method name. Save compiled kernel in target.
// Platform and build specific build options are automatically set.
void CLContext::buildKernel(cl::Kernel &target, std::string fileName, std::string methodName)
{
    // Kernel already exists
    if (target()) return;

    // Show progress
    window->showMessage("Building kernel", fileName);

    // Define build options to check cached kernel validity
    std::string buildOpts = "-DGPU -I./src -cl-denorms-are-zero -cl-fast-relaxed-math";
    Settings &s = Settings::getInstance();
    if (s.getUseBitstack()) buildOpts += " -DUSE_BITSTACK";
    if (s.getUseSoA()) buildOpts += " -DUSE_SOA";
    if (platformIsNvidia(platform)) buildOpts += " -cl-nv-verbose";
    #ifdef CPU_DEBUGGING
        buildOpts += " -g -s \"" + getAbsolutePath("src/" + fileName) + "\"";
    #endif
    
    cl::Program program;

    // CPU debugging segfaults if trying to use cached kernel!
    // Also need to let the driver do the include handling
    #ifdef CPU_DEBUGGING
        kernelFromSource("src/" + fileName, context, program, err);
        cl::vector<cl::Device> devices = { device };
        err = program.build(devices, buildOpts.c_str());

        // Check build log
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        if (buildLog.length() > 2)
            std::cout << "\n[" << fileName << " build log]:" << buildLog << std::endl;

        verify("Kernel compilation failed");
    #else
        // Build program using cache or sources
        program = kernelFromFile(fileName, buildOpts, platform, context, device, err);
        verify("Failed to create kernel program");
    #endif

    // Creating compute kernel from program
    target = cl::Kernel(program, methodName.c_str(), &err);
    verify("Failed to create compute kernel!");
}

void CLContext::setupPickKernel()
{
    buildKernel(kernel_pick, "kernel_pick.cl", "pick");
    int i = 0;
    err = 0;
    err |= kernel_pick.setArg(i++, renderParams);
    err |= kernel_pick.setArg(i++, triangleBuffer);
    err |= kernel_pick.setArg(i++, nodeBuffer);
    err |= kernel_pick.setArg(i++, indexBuffer);
    err |= kernel_pick.setArg(i++, pickResult);
    verify("Failed to set kernel_pick arguments!");
}

void CLContext::setupWfExtKernel()
{
    buildKernel(wf_extension, "wf_extrays.cl", "traceExtension");
    int i = 0;
    err = 0;
    err |= wf_extension.setArg(i++, tasksBuffer);
    err |= wf_extension.setArg(i++, queueCounters);
    err |= wf_extension.setArg(i++, extensionQueue);
    err |= wf_extension.setArg(i++, triangleBuffer);
    err |= wf_extension.setArg(i++, nodeBuffer);
    err |= wf_extension.setArg(i++, indexBuffer);
    err |= wf_extension.setArg(i++, renderParams);
    err |= wf_extension.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_extension arguments!");
}

void CLContext::setupWfLogicKernel()
{
    buildKernel(wf_logic, "wf_logic.cl", "logic");
    int i = 0;
    err = 0;
    err |= wf_logic.setArg(i++, tasksBuffer);
    err |= wf_logic.setArg(i++, pixelBuffer);
    err |= wf_logic.setArg(i++, queueCounters);
    err |= wf_logic.setArg(i++, extensionQueue);
    err |= wf_logic.setArg(i++, shadowQueue);
    err |= wf_logic.setArg(i++, raygenQueue);
    err |= wf_logic.setArg(i++, diffuseMatQueue);
    err |= wf_logic.setArg(i++, glossyMatQueue);
    err |= wf_logic.setArg(i++, ggxReflMatQueue);
    err |= wf_logic.setArg(i++, ggxRefrMatQueue);
    err |= wf_logic.setArg(i++, deltaMatQueue);
    err |= wf_logic.setArg(i++, triangleBuffer);
    err |= wf_logic.setArg(i++, nodeBuffer);
    err |= wf_logic.setArg(i++, indexBuffer);
    err |= wf_logic.setArg(i++, environmentMap);
    err |= wf_logic.setArg(i++, probTable);
    err |= wf_logic.setArg(i++, aliasTable);
    err |= wf_logic.setArg(i++, pdfTable);
    err |= wf_logic.setArg(i++, materialBuffer);
    err |= wf_logic.setArg(i++, texDataBuffer);
    err |= wf_logic.setArg(i++, texDescriptorBuffer);
    err |= wf_logic.setArg(i++, renderParams);
    err |= wf_logic.setArg(i++, NUM_TASKS);
    err |= wf_logic.setArg(i++, (cl_uint)false);
    verify("Failed to set wf_logic arguments!");
}

void CLContext::setupWfShadowKernel()
{
    buildKernel(wf_shadow, "wf_shadowrays.cl", "traceShadow");
    int i = 0;
    err = 0;
    err |= wf_shadow.setArg(i++, tasksBuffer);
    err |= wf_shadow.setArg(i++, queueCounters);
    err |= wf_shadow.setArg(i++, shadowQueue);
    err |= wf_shadow.setArg(i++, triangleBuffer);
    err |= wf_shadow.setArg(i++, nodeBuffer);
    err |= wf_shadow.setArg(i++, indexBuffer);
    err |= wf_shadow.setArg(i++, renderParams);
    err |= wf_shadow.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_shadow arguments!");
}

void CLContext::setupWfRaygenKernel()
{
    buildKernel(wf_raygen, "wf_raygen.cl", "genRays");
    int i = 0;
    err = 0;
    err |= wf_raygen.setArg(i++, tasksBuffer);
    err |= wf_raygen.setArg(i++, renderParams);
    err |= wf_raygen.setArg(i++, queueCounters);
    err |= wf_raygen.setArg(i++, raygenQueue);
    err |= wf_raygen.setArg(i++, extensionQueue);
    err |= wf_raygen.setArg(i++, currentPixelIdx);
    err |= wf_raygen.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_raygen arguments!");
}

void CLContext::setupWfDiffuseKernel()
{
    buildKernel(wf_diffuse, "wf_mat_diffuse.cl", "wavefrontDiffuse");
    int i = 0;
    err = 0;
    err |= wf_diffuse.setArg(i++, tasksBuffer);
    err |= wf_diffuse.setArg(i++, queueCounters);
    err |= wf_diffuse.setArg(i++, diffuseMatQueue);
    err |= wf_diffuse.setArg(i++, extensionQueue);
    err |= wf_diffuse.setArg(i++, materialBuffer);
    err |= wf_diffuse.setArg(i++, texDataBuffer);
    err |= wf_diffuse.setArg(i++, texDescriptorBuffer);
    err |= wf_diffuse.setArg(i++, renderParams);
    err |= wf_diffuse.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_diffuse arguments!");
}

void CLContext::setupWfGlossyKernel()
{
    buildKernel(wf_glossy, "wf_mat_glossy.cl", "wavefrontGlossy");
    int i = 0;
    err = 0;
    err |= wf_glossy.setArg(i++, tasksBuffer);
    err |= wf_glossy.setArg(i++, queueCounters);
    err |= wf_glossy.setArg(i++, glossyMatQueue);
    err |= wf_glossy.setArg(i++, extensionQueue);
    err |= wf_glossy.setArg(i++, materialBuffer);
    err |= wf_glossy.setArg(i++, texDataBuffer);
    err |= wf_glossy.setArg(i++, texDescriptorBuffer);
    err |= wf_glossy.setArg(i++, renderParams);
    err |= wf_glossy.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_glossy arguments!");
}

void CLContext::setupWfGGXReflKernel()
{
    buildKernel(wf_ggx_refl, "wf_mat_ggx_reflection.cl", "wavefrontGGXReflection");
    int i = 0;
    err = 0;
    err |= wf_ggx_refl.setArg(i++, tasksBuffer);
    err |= wf_ggx_refl.setArg(i++, queueCounters);
    err |= wf_ggx_refl.setArg(i++, ggxReflMatQueue);
    err |= wf_ggx_refl.setArg(i++, extensionQueue);
    err |= wf_ggx_refl.setArg(i++, materialBuffer);
    err |= wf_ggx_refl.setArg(i++, texDataBuffer);
    err |= wf_ggx_refl.setArg(i++, texDescriptorBuffer);
    err |= wf_ggx_refl.setArg(i++, renderParams);
    err |= wf_ggx_refl.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_ggx_refl arguments!");
}

void CLContext::setupWfGGXRefrKernel()
{
    buildKernel(wf_ggx_refr, "wf_mat_ggx_refraction.cl", "wavefrontGGXRefraction");
    int i = 0;
    err = 0;
    err |= wf_ggx_refr.setArg(i++, tasksBuffer);
    err |= wf_ggx_refr.setArg(i++, queueCounters);
    err |= wf_ggx_refr.setArg(i++, ggxRefrMatQueue);
    err |= wf_ggx_refr.setArg(i++, extensionQueue);
    err |= wf_ggx_refr.setArg(i++, materialBuffer);
    err |= wf_ggx_refr.setArg(i++, texDataBuffer);
    err |= wf_ggx_refr.setArg(i++, texDescriptorBuffer);
    err |= wf_ggx_refr.setArg(i++, renderParams);
    err |= wf_ggx_refr.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_ggx_refr arguments!");
}

void CLContext::setupWfDeltaKernel()
{
    buildKernel(wf_delta, "wf_mat_delta.cl", "wavefrontDelta");
    int i = 0;
    err = 0;
    err |= wf_delta.setArg(i++, tasksBuffer);
    err |= wf_delta.setArg(i++, queueCounters);
    err |= wf_delta.setArg(i++, deltaMatQueue);
    err |= wf_delta.setArg(i++, extensionQueue);
    err |= wf_delta.setArg(i++, materialBuffer);
    err |= wf_delta.setArg(i++, texDataBuffer);
    err |= wf_delta.setArg(i++, texDescriptorBuffer);
    err |= wf_delta.setArg(i++, renderParams);
    err |= wf_delta.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_delta arguments!");
}

void CLContext::setupResetKernel()
{
	buildKernel(mk_reset, "mk_reset.cl", "reset");

	// Set initial kernel params
	int i = 0;
	err = 0;
	err |= mk_reset.setArg(i++, tasksBuffer);
	err |= mk_reset.setArg(i++, pixelBuffer);
	err |= mk_reset.setArg(i++, renderParams);
	err |= mk_reset.setArg(i++, NUM_TASKS);
	verify("Failed to set mk_reset arguments!");
}

void CLContext::setupWfResetKernel()
{
    buildKernel(wf_reset, "wf_reset.cl", "reset");

    int i = 0;
    err = 0;
    err |= wf_reset.setArg(i++, tasksBuffer);
    err |= wf_reset.setArg(i++, pixelBuffer);
    err |= wf_reset.setArg(i++, queueCounters);
    err |= wf_reset.setArg(i++, raygenQueue);
    err |= wf_reset.setArg(i++, renderParams);
    err |= wf_reset.setArg(i++, NUM_TASKS);
    verify("Failed to set wf_reset arguments!");
}

void CLContext::setupRayGenKernel()
{
    buildKernel(mk_raygen, "mk_raygen.cl", "genCameraRays");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_raygen.setArg(i++, tasksBuffer);
    err |= mk_raygen.setArg(i++, renderParams);
    err |= mk_raygen.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_raygen arguments!");
}

void CLContext::setupNextVertexKernel()
{
    buildKernel(mk_next_vertex, "mk_next_vertex.cl", "nextVertex");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_next_vertex.setArg(i++, tasksBuffer);
    err |= mk_next_vertex.setArg(i++, materialBuffer);
    err |= mk_next_vertex.setArg(i++, texDataBuffer);
    err |= mk_next_vertex.setArg(i++, texDescriptorBuffer);
    err |= mk_next_vertex.setArg(i++, triangleBuffer);
    err |= mk_next_vertex.setArg(i++, nodeBuffer);
    err |= mk_next_vertex.setArg(i++, indexBuffer);
    err |= mk_next_vertex.setArg(i++, renderParams);
    err |= mk_next_vertex.setArg(i++, renderStats);
    err |= mk_next_vertex.setArg(i++, environmentMap);
	err |= mk_next_vertex.setArg(i++, pdfTable);
    err |= mk_next_vertex.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_next_vertex arguments!");
}

void CLContext::setupBsdfSampleKernel()
{
    buildKernel(mk_sample_bsdf, "mk_sample_bsdf.cl", "sampleBsdf");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_sample_bsdf.setArg(i++, tasksBuffer);
    err |= mk_sample_bsdf.setArg(i++, materialBuffer);
    err |= mk_sample_bsdf.setArg(i++, texDataBuffer);
    err |= mk_sample_bsdf.setArg(i++, texDescriptorBuffer);
    err |= mk_sample_bsdf.setArg(i++, environmentMap);
    err |= mk_sample_bsdf.setArg(i++, probTable);
    err |= mk_sample_bsdf.setArg(i++, aliasTable);
	err |= mk_sample_bsdf.setArg(i++, pdfTable);
    err |= mk_sample_bsdf.setArg(i++, triangleBuffer);
    err |= mk_sample_bsdf.setArg(i++, nodeBuffer);
    err |= mk_sample_bsdf.setArg(i++, indexBuffer);
    err |= mk_sample_bsdf.setArg(i++, renderParams);
    err |= mk_sample_bsdf.setArg(i++, renderStats);
    err |= mk_sample_bsdf.setArg(i++, NUM_TASKS);
    err |= mk_sample_bsdf.setArg(i++, 0);
    verify("Failed to set mk_sample_bsdf arguments!");
}

void CLContext::setupSplatKernel()
{
    buildKernel(mk_splat, "mk_splat.cl", "splat");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_splat.setArg(i++, tasksBuffer);
	err |= mk_splat.setArg(i++, pixelBuffer);
    err |= mk_splat.setArg(i++, renderParams);
    err |= mk_splat.setArg(i++, renderStats);
    err |= mk_splat.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_splat arguments!");
}

void CLContext::setupSplatPreviewKernel()
{
    buildKernel(mk_splat_preview, "mk_splat_preview.cl", "splatPreview");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_splat_preview.setArg(i++, tasksBuffer);
    err |= mk_splat_preview.setArg(i++, pixelBuffer);
    err |= mk_splat_preview.setArg(i++, renderParams);
    err |= mk_splat_preview.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_splat_preview arguments!");
}

void CLContext::setupPostprocessKernel()
{
    buildKernel(mk_postprocess, "mk_postprocess.cl", "process");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_postprocess.setArg(i++, pixelBuffer); // raw pixels
    err |= mk_postprocess.setArg(i++, sharedMemory.back()); // preview
    err |= mk_postprocess.setArg(i++, renderParams);
    err |= mk_postprocess.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_postprocess arguments!");
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;
}

void CLContext::setupPixelStorage(PTWindow *window)
{
    if (sharedMemory.size() > 0)
    {
        sharedMemory.clear(); // memory freed by cl-cpp-wrapper
    }

    GLuint *tex_arr = window->getTexPtr();
    GLuint gl_PBO = window->getPBO();
    unsigned int numPixels = window->getTexWidth() * window->getTexHeight();

    pixelBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(cl_float) * 4, NULL, &err); // microkernel pixel buffer
    previewBuffer = cl::BufferGL(context, CL_MEM_READ_WRITE, gl_PBO, &err); // GL preview buffer
    sharedMemory = { previewBuffer };
    verify("CL pixel storage creation failed!");

	// Set new kernel args (pointers might have changed)
	// Megakernel args reset anyway due to ping pong
	err = 0;
	if (mk_reset())
		err |= mk_reset.setArg(1, pixelBuffer);
	if (mk_splat())
		err |= mk_splat.setArg(1, pixelBuffer);
    if (mk_splat_preview())
        err |= mk_splat_preview.setArg(1, pixelBuffer);
    if (wf_logic())
        err |= wf_logic.setArg(1, pixelBuffer);
    if (wf_reset())
        err |= wf_reset.setArg(1, pixelBuffer);
    if (mk_postprocess())
    {
        err |= mk_postprocess.setArg(0, pixelBuffer);
        err |= mk_postprocess.setArg(1, previewBuffer);
    }
        
	verify("Failed to update kernel pixel storage args");
}

void CLContext::saveImage(std::string filename, const RenderParams &params)
{
    unsigned int numBytes = params.width * params.height * 3; // rgb
    unsigned int numFloats = params.width * params.height * 4; // rgba
	std::unique_ptr<unsigned char[]> dataBytes(new unsigned char[numBytes]);
    std::unique_ptr<float[]> dataFloats(new float[numFloats]);

    bool hdr = endsWith(filename, ".hdr") || endsWith(filename, ".HDR");

    // Copy data to host
    err = 0;
    err |= cmdQueue.enqueueAcquireGLObjects(&sharedMemory);

    cl::Buffer &pixels = (hdr) ? pixelBuffer : previewBuffer;
    err |= cmdQueue.enqueueReadBuffer(pixels, CL_TRUE, 0, numFloats * sizeof(float), dataFloats.get());
    err |= cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    err |= cmdQueue.finish();
    verify("Failed to copy pixel buffer to host!");
    
    if (hdr)
    {
        // Save linear unclamped values
        for (int i = 0; i < numFloats; i += 4)
        {
            
            float *r = &dataFloats[i] + 0;
            float *g = &dataFloats[i] + 1;
            float *b = &dataFloats[i] + 2;
            float *a = &dataFloats[i] + 3;

            *r /= *a;
            *g /= *a;
            *b /= *a;
            *a = 1.0f;
        }

        ILuint imageID = ilGenImage();
        ilBindImage(imageID);
        ilTexImage(params.width, params.height, 1, 4, IL_RGBA, IL_FLOAT, dataFloats.get());
        ilSaveImage(filename.c_str());
        ilDeleteImage(imageID);
    }
    else
    {
        // Convert to bytes
        // Already tonemapped and gamma-corrected
        int counter = 0;
        for (int i = 0; i < numFloats; i += 4)
        {
            float r = dataFloats[i + 0];
            float g = dataFloats[i + 1];
            float b = dataFloats[i + 2];
            float a = dataFloats[i + 3];

            // Convert to bytes
            auto clamp = [](float value) { return std::max(0.0f, std::min(1.0f, value)); };
            dataBytes[counter++] = (unsigned char)(255 * clamp(r));
            dataBytes[counter++] = (unsigned char)(255 * clamp(g));
            dataBytes[counter++] = (unsigned char)(255 * clamp(b));
        }

        ILuint imageID = ilGenImage();
        ilBindImage(imageID);
        ilTexImage(params.width, params.height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, dataBytes.get());
        ilSaveImage(filename.c_str());
        ilDeleteImage(imageID);
    }
    

    // Check for errors
    ILenum Error = IL_NO_ERROR;
    while ((Error = ilGetError()) != IL_NO_ERROR)
    {
        printf("\n%d: %s", Error, iluErrorString(Error));
    }
	
	std::cout << ((Error == IL_NO_ERROR) ? "\nSaved " : "\nFailed saving ") << filename << std::endl;
}

void CLContext::createEnvMap(EnvironmentMap *map)
{
	int width = map->getWidth(), height = map->getHeight();
	float *data = map->getData();

    // Convert rgb to rgba (OpenCL doesn't support floats for RGB-images)
    float *rgba = new float[width * height * 4];
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // RGB
            for (int c = 0; c < 3; c++) // color channels
            {
                rgba[(h * width + w) * 4 + c] = data[(h * width + w) * 3 + c];
            }
            // Alpha
            rgba[(h * width + w) * 4 + 3] = 1.0f;
        }
    }

	// Upload rgb colors
    const cl::ImageFormat format(CL_RGBA, CL_FLOAT);
    environmentMap = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width, height, 0, rgba, &err);
    verify("Environment map creation failed!");

	// Upload probability and alias tables for importance sampling
	size_t pBytes = width * height * sizeof(float);
	size_t aBytes = width * height * sizeof(int);
	probTable = cl::Buffer(context, CL_MEM_READ_ONLY, pBytes, NULL, &err);
	aliasTable = cl::Buffer(context, CL_MEM_READ_ONLY, aBytes, NULL, &err);
	pdfTable = cl::Buffer(context, CL_MEM_READ_ONLY, pBytes, NULL, &err);
	verify("Env map IS table creation failed");

	err |= cmdQueue.enqueueWriteBuffer(probTable, CL_TRUE, 0, pBytes, map->getProbTable());
	err |= cmdQueue.enqueueWriteBuffer(aliasTable, CL_TRUE, 0, aBytes, map->getAliasTable());
	err |= cmdQueue.enqueueWriteBuffer(pdfTable, CL_TRUE, 0, pBytes, map->getPdfTable());
	verify("Env map IS table writing failed");

	// Cleanup
    delete[] rgba;

    // Update env map references
    setupKernels();
}

void CLContext::setupScene()
{
    // Lights
    size_t l_bytes = sizeof(test_lights);
    lightBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, l_bytes, NULL, &err);
    verify("Light buffer creation failed!");

    err = cmdQueue.enqueueWriteBuffer(lightBuffer, CL_TRUE, 0, l_bytes, test_lights);
    verify("Light buffer writing failed!");

	// Dummy env map
	float rgba[4] { 0.0f, 0.0f, 0.0f, 0.0f };
	environmentMap = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_FLOAT), 1, 1, 0, rgba, &err);
	verify("Dummy env map creation failed");

    std::cout << "Scene initialization succeeded!" << std::endl;
}

// Upload BVH data, geometry and materials to GPU
void CLContext::uploadSceneData(BVH *bvh, Scene *scene)
{
    std::vector<RTTriangle> *tris = bvh->m_triangles;
    std::vector<cl_uint> *indices = &bvh->m_indices; 
    std::vector<Node> *nodes = &bvh->m_nodes;
    std::vector<Material> *materials = &scene->getMaterials();

    size_t t_bytes = tris->size() * sizeof(RTTriangle);
    size_t i_bytes = indices->size() * sizeof(cl_uint);
    size_t n_bytes = nodes->size() * sizeof(Node);
    size_t m_bytes = materials->size() * sizeof(Material);

    // Allocate memory for buffers
    triangleBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, t_bytes, NULL, &err);
    verify("Triangle buffer creation failed!");

    indexBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, i_bytes, NULL, &err);
    verify("Index buffer creation failed!");

    nodeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, n_bytes, NULL, &err);
    verify("Node buffer creation failed!");

    if(m_bytes > 0) materialBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, m_bytes, NULL, &err);
    verify("Material buffer creation failed!");


    // Write data to buffers
    err = cmdQueue.enqueueWriteBuffer(triangleBuffer, CL_TRUE, 0, t_bytes, tris->data());
    verify("Triangle buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, i_bytes, indices->data());
    verify("Index buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(nodeBuffer, CL_TRUE, 0, n_bytes, nodes->data());
    verify("Node buffer writing failed!");

    if(m_bytes > 0) err = cmdQueue.enqueueWriteBuffer(materialBuffer, CL_TRUE, 0, m_bytes, materials->data());
    verify("Material buffer writing failed!");

    // Pack texture data into aggregate array
    packTextures(scene);

    // Ensures that the kernels have the correct arguments
    setupKernels();
}

// Upload texture data to GPU
// Avoids intermediate buffers to keep RAM usage low
void CLContext::packTextures(Scene *scene)
{
    std::vector<Texture*> textures = scene->getTextures();

    if (textures.size() == 0) return;
    
    // Calculate total size required for texture data
    size_t t_bytes = 0;
    for (Texture *tex : textures)
    {
        t_bytes += tex->getWidth() * tex->getHeight() * 4 * 1; // RGBA
    }

    // Create buffers for texture data & descriptors
    size_t d_bytes = textures.size() * sizeof(TexDescriptor);
    texDescriptorBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, d_bytes, NULL, &err);
    verify("Texture descriptor buffer creation failed!");
    texDataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, t_bytes, NULL, &err);
    verify("Texture data buffer creation failed!");
    
    // Upload data, create descriptors
    std::vector<TexDescriptor> descs;
    cl_uint offset = 0;
    for (Texture *tex : textures)
    {
        TexDescriptor desc;
        desc.offset = offset;
        desc.width = tex->getWidth();
        desc.height = tex->getHeight();
        descs.push_back(desc);

        cl_uint len = tex->getWidth() * tex->getHeight() * 4 * 1; // RGBA
        err = cmdQueue.enqueueWriteBuffer(texDataBuffer, CL_TRUE, offset, len, tex->getData());
        verify("Texture data buffer writing failed!");

        offset += len;
    }

    // Upload descriptors
    err = cmdQueue.enqueueWriteBuffer(texDescriptorBuffer, CL_TRUE, 0, d_bytes, descs.data());
    verify("Texture descriptor buffer writing failed!");
}

// Passing structs to kernels is broken in several drivers (e.g. GT 750M on MacOS)
// Allocating memory for the rendering params is more compatible
void CLContext::setupParams()
{
    renderParams = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(RenderParams) * 1, NULL, &err);
    verify("Params buffer creation failed!");
}

void CLContext::setupPickResult()
{
    pickResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(Hit) * 1, NULL, &err);
    verify("Pick result creation failed!");
}

void CLContext::setupStats()
{
    renderStats = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(RenderStats) * 1, NULL, &err);
    verify("RenderStats creation failed!");
    resetStats();
}

void CLContext::resetStats()
{
    RenderStats s = { 0, 0, 0, 0 };
    statsAsync = s;
    err = cmdQueue.enqueueWriteBuffer(renderStats, CL_TRUE, 0, sizeof(RenderStats), &s);
    verify("Stats buffer reset failed!");
}

void CLContext::fetchStatsAsync()
{
    err = cmdQueue.enqueueReadBuffer(renderStats, CL_FALSE, 0, sizeof(RenderStats), &statsAsync);
    verify("Failed to enqueue async stat transfer!");
}

void CLContext::updateRenderPerf(float deltaT)
{
    double scale = 1e6 * deltaT;
    renderPerf.primary = statsAsync.primaryRays / scale;
    renderPerf.extension = statsAsync.extensionRays / scale;
    renderPerf.shadow = statsAsync.shadowRays / scale;
    renderPerf.samples = statsAsync.samples / scale;
    renderPerf.total = renderPerf.primary + renderPerf.extension + renderPerf.shadow;
}

const PerfNumbers CLContext::getRenderPerf()
{
    return renderPerf;
}

const RenderStats CLContext::getStats()
{
    return statsAsync;
}

void CLContext::enqueueGetCounters(QueueCounters *cnt)
{
    err = cmdQueue.enqueueReadBuffer(queueCounters, CL_FALSE, 0, 1 * sizeof(QueueCounters), cnt);
}

void CLContext::checkTracingPerf()
{
    // Check ray tracing perf without overhead
    cl_ulong t0Ext, t0Shadow;
    cl_ulong t1Ext, t1Shadow;

    clGetEventProfilingInfo(extRayEvent(), CL_PROFILING_COMMAND_START, sizeof(t0Ext), &t0Ext, NULL);
    clGetEventProfilingInfo(extRayEvent(), CL_PROFILING_COMMAND_END, sizeof(t1Ext), &t1Ext, NULL);
    clGetEventProfilingInfo(shdwRayEvent(), CL_PROFILING_COMMAND_START, sizeof(t0Shadow), &t0Shadow, NULL);
    clGetEventProfilingInfo(shdwRayEvent(), CL_PROFILING_COMMAND_END, sizeof(t1Shadow), &t1Shadow, NULL);

    // Extension rays
    double timeNs = t1Ext - t0Ext;
    double timeMs = timeNs / 1000000.0;
    double timeS = timeMs / 1000.0;

    double scale = 1e6 * timeS;
    double MRaysExt = statsAsync.extensionRays / scale;
    printf("Ext ray time: %0.3f milliseconds, speed: %.2f MRays/s \n", timeMs, MRaysExt);

    // Shadow rays
    timeNs = t1Shadow - t0Shadow;
    timeMs = timeNs / 1000000.0;
    timeS = timeMs / 1000.0;

    scale = 1e6 * timeS;
    double MRaysShadow = statsAsync.shadowRays / scale;
    printf("Shadow ray time: %0.3f milliseconds, speed: %.2f MRays/s \n", timeMs, MRaysShadow);
}

void CLContext::updateParams(const RenderParams &params)
{
    err = cmdQueue.enqueueWriteBuffer(renderParams, CL_FALSE, 0, sizeof(RenderParams), &params);
    verify("RenderParam writing failed");
}

void CLContext::enqueueResetKernel(const RenderParams &params)
{
	err = 0;
	err |= cmdQueue.enqueueNDRangeKernel(mk_reset, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
	verify("Failed to enqueue reset kernel!");
}

void CLContext::enqueueRayGenKernel(const RenderParams &params)
{
    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_raygen, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue ray gen kernel!");
}

void CLContext::enqueueNextVertexKernel(const RenderParams &params)
{
    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_next_vertex, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue next vertex kernel!");
}

void CLContext::enqueueBsdfSampleKernel(const RenderParams &params, const cl_uint iteration)
{
    // Enqueue 1D range
    err = 0;
    err |= mk_sample_bsdf.setArg(14, iteration);
    err = cmdQueue.enqueueNDRangeKernel(mk_sample_bsdf, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue bsdf sample kernel!");
}

void CLContext::enqueueSplatKernel(const RenderParams &params, const cl_uint iteration)
{
    err = 0;
    err |= mk_splat.setArg(5, iteration);
    verify("Failed to set mk_splat arguments!");

    // TODO: find out why my GTX 780 won't enqueue 1D kernels! (due to image2d_type?)
    // TODO: also, look at having global wg be a multiple of local wg (or a multiple of 32/64)
    err = cmdQueue.enqueueNDRangeKernel(mk_splat, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat kernel!");
}

void CLContext::enqueueSplatPreviewKernel(const RenderParams &params)
{
    err = cmdQueue.enqueueNDRangeKernel(mk_splat_preview, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat preview kernel!");
}

void CLContext::enqueuePostprocessKernel(const RenderParams & params)
{   
    std::vector<cl::Memory> previewBuffer{ sharedMemory.back() };
    err = cmdQueue.enqueueAcquireGLObjects(&previewBuffer);
    verify("Failed to enqueue GL object acquisition!");

    // 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_postprocess, cl::NullRange, cl::NDRange(params.width * params.height), cl::NullRange);
    verify("Failed to enqueue postprocess kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&previewBuffer);
    verify("Failed to enqueue GL object release!");
}

void CLContext::enqueueWfResetKernel(const RenderParams & params)
{
    cl_uint numElems = std::max(NUM_TASKS, params.width * params.height);
    err = cmdQueue.enqueueNDRangeKernel(wf_reset, cl::NullRange, cl::NDRange(numElems), cl::NullRange);
    verify("Failed to enqueue wf_reset");
}

void CLContext::enqueueWfRaygenKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_raygen, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_raygen");
}

void CLContext::enqueueWfExtRayKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_extension, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange, 0, &extRayEvent);
    verify("Failed to enqueue wf_extension");
}

void CLContext::enqueueWfShadowRayKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_shadow, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange, 0, &shdwRayEvent);
    verify("Failed to enqueue wf_shadow");
}

void CLContext::enqueueWfLogicKernel(const bool firstIteration)
{
    err |= wf_logic.setArg(23, (cl_uint)firstIteration);
    err |= cmdQueue.enqueueNDRangeKernel(wf_logic, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_logic");
}

void CLContext::enqueueWfMaterialKernels(const RenderParams & params)
{
    enqueueWfDiffuseKernel(params);
    enqueueWfGlossyKernel(params);
    enqueueWfGGXReflKernel(params);
    enqueueWfGGXRefrKernel(params);
    enqueueWfDeltaKernel(params);
}

void CLContext::enqueueWfDiffuseKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_diffuse, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_diffuse");
}

void CLContext::enqueueWfGlossyKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_glossy, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_glossy");
}

void CLContext::enqueueWfGGXReflKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_ggx_refl, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_ggx_refl");
}

void CLContext::enqueueWfGGXRefrKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_ggx_refr, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_ggx_refr");
}

void CLContext::enqueueWfDeltaKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_delta, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_delta");
}


// Clear wavefront queues by setting counters to zero
void CLContext::enqueueClearWfQueues()
{
    QueueCounters empty = {};
    hostCounters = empty;
    err = cmdQueue.enqueueWriteBuffer(queueCounters, CL_FALSE, 0, sizeof(QueueCounters), &hostCounters);
    verify("Failed to enqueue wavefront queueCounter read");
}

void CLContext::finishQueue()
{
    err = cmdQueue.finish();
    verify("Failed to finish command queue!");
}

void CLContext::updatePixelIndex(cl_uint numPixels, cl_uint numNewPaths)
{
    pixelIdx = (pixelIdx + numNewPaths) % numPixels;
    err = cmdQueue.enqueueWriteBuffer(currentPixelIdx, CL_FALSE, 0, sizeof(cl_uint), &pixelIdx); // will be available when raygen runs
}

void CLContext::resetPixelIndex()
{
    pixelIdx = 0;
    err = cmdQueue.enqueueWriteBuffer(currentPixelIdx, CL_FALSE, 0, sizeof(cl_uint), &pixelIdx);
}

Hit CLContext::pickSingle(float NDCx, float NDCy)
{
    err = 0;
    err |= kernel_pick.setArg(5, NDCx);
    err |= kernel_pick.setArg(6, NDCy);
    verify("Failed to set pick kernel coordinates");

    Hit hit;

    err |= cmdQueue.enqueueNDRangeKernel(kernel_pick, cl::NullRange, cl::NDRange(1), cl::NullRange);
    err |= cmdQueue.enqueueReadBuffer(pickResult, CL_FALSE, 0, 1 * sizeof(Hit), &hit);
    cmdQueue.finish();
    verify("Failed to execute pick kernel or get result");

    return hit;
}

cl::Platform &CLContext::getPlatformByName(std::vector<cl::Platform> &platforms, std::string name) {
    for (cl::Platform &p : platforms) {
        std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
        if (platformName.find(name) != std::string::npos) {
            return p;
        }
    }

    std::cout << "No platform name containing \"" << name << "\" found!" << std::endl;
    return platforms[0];
}

cl::Device &CLContext::getDeviceByName(std::vector<cl::Device> &devices, std::string name) {
    for (cl::Device &d : devices) {
        std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
        if (deviceName.find(name) != std::string::npos) {
            return d;
        }
    }

    std::cout << "No device name containing \"" << name << "\" in selected context!" << std::endl;
    return devices[0];
}

// Return info about error
std::string CLContext::errorString()
{
    const int SIZE = 64;
    std::string errors[SIZE] =
    {
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE", "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE", "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH", "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE",
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE", "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE", "CL_INVALID_CONTEXT", "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE", "CL_INVALID_HOST_PTR", "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER", "CL_INVALID_BINARY", "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME", "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX", "CL_INVALID_ARG_VALUE", "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE", "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET", "CL_INVALID_EVENT_WAIT_LIST", "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT", "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL", "CL_INVALID_GLOBAL_WORK_SIZE"
    };

    const int ind = -err;
    return (ind >= 0 && ind < SIZE) ? errors[ind] : "unknown!";
}

// Check error, second optional parameter acts as boolean predicate
void CLContext::verify(std::string msg, int pred)
{
	// Use default value if predicate is -1
	int code = (pred > -1) ? !pred : this->err;

    if(code != CL_SUCCESS)
    {
        std::string message = msg + " (" + errorString() + ")";
        std::cout << message << std::endl;
        waitExit();
    }
}

// Print the devices, C++ style
void CLContext::printDevices()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    const std::string DECORATOR = "================";

    int platform_id = 0;
    int device_id = 0;

    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    for(cl::Platform &platform : platforms)
    {
        std::cout << DECORATOR << " Platform " << platform_id++ << " (" << platform.getInfo<CL_PLATFORM_NAME>() << ") " << DECORATOR << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        for(cl::Device &device : devices)
        {
            bool GPU = (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);

            std::cout << "Device " << device_id++ << ": " << std::endl;
            std::cout << "\tName: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "\tType: " << (GPU ? "(GPU)" : "(CPU)") << std::endl;
            std::cout << "\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\tCompute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\tGlobal Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
            std::cout << "\tMax Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            std::cout << "\tMax Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            std::cout << "\tLocal Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            std::cout << "\tAvailable: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
        }
        std::cout << std::endl;
    }
}



