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
    // Remove kernel caching to always get build logs (on NVIDIA hardware)
    // The cache also IGNORES included files when comparing hashes, making it useless
    #ifdef _WIN32
        _putenv_s("CUDA_CACHE_DISABLE", "1");
    #endif

    // printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    platform = getPlatformByName(platforms, Settings::getInstance().getPlatformName());
    std::cout << "PLATFORM: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	#ifdef CPU_DEBUGGING
		platform.getDevices(CL_DEVICE_TYPE_CPU, &clDevices);
	#else
		platform.getDevices(CL_DEVICE_TYPE_GPU, &clDevices);
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

    context = cl::Context(clDevices, props, NULL, NULL, &err);
    verify("Failed to create shared context");

    // Select correct device from context based on settings
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    device = getDeviceByName(devices, Settings::getInstance().getDeviceName());
    std::cout << "DEVICE: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create command queue for context
    cmdQueue = cl::CommandQueue(context, device, 0, &err);
    verify("Failed to create command queue!");
}

void CLContext::setup(PTWindow *window)
{
    this->window = window;

    // Setup RenderParams
    setupParams();

    // Setup RenerStats
    setupStats();

    // Create OpenCL buffer from OpenGL PBO
    setupPixelStorage(window->getTexPtr(), window->getPBO());

    // Allocate device memory for scene
    setupScene();

    // Build kernels, set their params
    setupKernels();
}

void CLContext::setupKernels()
{
    initMCBuffers();
	setupResetKernel();
    setupRayGenKernel();
    setupNextVertexKernel();
    setupBsdfSampleKernel();
    setupSplatKernel();
    setupSplatPreviewKernel();
    setupMegaKernel();
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

    // Init tasks buffer
    GPUTaskState *initialTaskStates = new GPUTaskState[NUM_TASKS];
    
    // Initial task state
    auto getDefaultState = []()
    {
        GPUTaskState curr;
        curr.orig = float3(0.0f); // overwritten in raygen kernel
        curr.dir = float3(0.0f, 0.0f, -1.0f); // overwritten in raygen kernel
        curr.T = float3(1.0f);
        curr.Ei = float3(0.0f);
        curr.phase = MK_GENERATE_CAMERA_RAY;
        curr.pathLen = 0;
        curr.seed = (unsigned int)rand();
        curr.samples = 0;
        curr.lastSpecular = (cl_uint)true;
        curr.lastPdfW = 0.0f;

        return curr;
    };

    // Write initial state in SoA or AoS format
    if (Settings::getInstance().getUseSoA())
    {
        // Maximum coalescing: 32bit boundaries
        float *data = (float*)initialTaskStates;
        for (int i = 0; i < NUM_TASKS; i++)
        {
            // Initialize struct normally
            GPUTaskState curr = getDefaultState();

            float *curr_data = (float*)(&curr);
            for (int j = 0; j < sizeof(GPUTaskState) / sizeof(float); j++) // 32 bits at a time
            {
                data[j * NUM_TASKS + i] = curr_data[j];
            }
        }

        // GPUTaskState *recovered = new GPUTaskState[NUM_TASKS];
        // copyToHost(recovered, initialTaskStates, NUM_TASKS);
        // delete[] recovered;
    }
    else
    {
        std::for_each(initialTaskStates + 0, initialTaskStates + NUM_TASKS, [&](GPUTaskState &s) { s = getDefaultState(); });
    }
    
    err = cmdQueue.enqueueWriteBuffer(tasksBuffer, CL_TRUE, 0, t_bytes, initialTaskStates);
    delete[] initialTaskStates;
    verify("Task buffer writing failed!");

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

    // Read kernel source from file
    std::string kernelPath = "src/" + fileName;

    cl::Program program;
    kernelFromFile(kernelPath, context, program, err);
    verify("Failed to create compute program for " + fileName);

    // Build kernel source (create compute program)
    // Define "GPU" to disable cl-prefixed types in shared headers (cl_float4 => float4 etc.)
    std::string buildOpts = "-DGPU -I./src -cl-denorms-are-zero";
    if (platformIsNvidia(platform)) buildOpts += " -cl-nv-verbose";
    #ifdef CPU_DEBUGGING
        buildOpts += " -g -s \"" + getAbsolutePath(kernelPath) + "\"";
    #endif

    // Add bitstack and SoA toggles from settings
    Settings &s = Settings::getInstance();
    if (s.getUseBitstack()) buildOpts += " -DUSE_BITSTACK";
    if (s.getUseSoA()) buildOpts += " -DUSE_SOA";

    err = program.build(clDevices, buildOpts.c_str());
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

    // Check build log
    if (buildLog.length() > 2)
        std::cout << "\n[" << fileName << " build log]:" << buildLog << std::endl;
    verify("Failed to build compute program!");

    // Creating compute kernel from program
    target = cl::Kernel(program, methodName.c_str(), &err);
    verify("Failed to create compute kernel!");

	// Read program binary (NVIDIA: PTX)
	auto ptxs = program.getInfo<CL_PROGRAM_BINARIES>();
	std::vector<unsigned char> ptx = ptxs[0];

	// Open target file in overwrite-mode
	std::ofstream stream;
	std::string type = (platformIsNvidia(platform)) ? ".ptx" : ".bin";
	stream.open("data/kernel_binaries/" + fileName + type, std::ofstream::out | std::ofstream::trunc);
	verify("Failed to open kernel binary file", stream.good());
	
	// Write binary to file
	stream << ptx.data() << std::endl;
	verify("Failed to write kernel binary", stream.good());
	stream.close();
	ptx.clear();
}

void CLContext::setupMegaKernel()
{
    buildKernel(kernel_monolith, "kernel_monolith.cl", "trace");

    int i = 0;
    err = 0;
    err |= kernel_monolith.setArg(i++, sharedMemory[1]); // src
    err |= kernel_monolith.setArg(i++, sharedMemory[0]); // dst
    err |= kernel_monolith.setArg(i++, texDataBuffer);
    err |= kernel_monolith.setArg(i++, texDescriptorBuffer);
    err |= kernel_monolith.setArg(i++, lightBuffer);
    err |= kernel_monolith.setArg(i++, triangleBuffer);
    err |= kernel_monolith.setArg(i++, materialBuffer);
    err |= kernel_monolith.setArg(i++, nodeBuffer);
    err |= kernel_monolith.setArg(i++, indexBuffer);
    err |= kernel_monolith.setArg(i++, environmentMap);
    err |= kernel_monolith.setArg(i++, probTable);
    err |= kernel_monolith.setArg(i++, aliasTable);
    err |= kernel_monolith.setArg(i++, pdfTable);
    err |= kernel_monolith.setArg(i++, renderParams);
    err |= kernel_monolith.setArg(i++, renderStats);
    err |= kernel_monolith.setArg(i++, 0); // iteration
    verify("Failed to set kernel arguments!");
}

void CLContext::setupResetKernel()
{
	buildKernel(mk_reset, "mk_reset.cl", "reset");

	// Set initial kernel params
	int i = 0;
	err = 0;
	err |= mk_reset.setArg(i++, tasksBuffer);
	err |= mk_reset.setArg(i++, sharedMemory.back());
	err |= mk_reset.setArg(i++, renderParams);
	err |= mk_reset.setArg(i++, NUM_TASKS);
	verify("Failed to set mk_reset arguments!");
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
	err |= mk_splat.setArg(i++, sharedMemory.back()); // pixel buffer
    err |= mk_splat.setArg(i++, renderParams);
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
    err |= mk_splat_preview.setArg(i++, sharedMemory.back()); // pixel buffer
    err |= mk_splat_preview.setArg(i++, renderParams);
    err |= mk_splat_preview.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_splat_preview arguments!");
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;
}

void CLContext::setupPixelStorage(GLuint *tex_arr, GLuint gl_PBO)
{
    if (sharedMemory.size() > 0)
    {
        sharedMemory.clear(); // memory freed by cl-cpp-wrapper
    }

    frontBuffer = cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex_arr[0], &err);
    backBuffer = cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex_arr[1], &err);
    pixelBuffer = cl::BufferGL(context, CL_MEM_READ_WRITE, gl_PBO, &err); // microkernel pixel buffer
    sharedMemory = { frontBuffer, backBuffer, pixelBuffer };
    verify("CL pixel storage creation failed!");

	// Set new kernel args (pointers might have changed)
	// Megakernel args reset anyway due to ping pong
	err = 0;
	if (mk_reset())
		err |= mk_reset.setArg(1, sharedMemory.back());
	if (mk_splat())
		err |= mk_splat.setArg(1, sharedMemory.back());
    if (mk_splat_preview())
        err |= mk_splat_preview.setArg(1, sharedMemory.back());
	verify("Failed to update kernel pixel storage args");
}

void CLContext::saveImage(std::string filename, const RenderParams &params, bool usingMicroKernel)
{
    unsigned int numBytes = params.width * params.height * 3; // rgb
    unsigned int numFloats = params.width * params.height * 4; // rgba
	std::unique_ptr<unsigned char[]> dataBytes(new unsigned char[numBytes]);
    std::unique_ptr<float[]> dataFloats(new float[numFloats]); 

    // Copy data to host
    err = 0;
    err |= cmdQueue.enqueueAcquireGLObjects(&sharedMemory);

    if (usingMicroKernel)
    {
        err |= cmdQueue.enqueueReadBuffer(pixelBuffer, CL_TRUE, 0, numFloats * sizeof(float), dataFloats.get());
    }
    else
    {
        std::array<cl::size_type, 3> orig = { 0, 0, 0 };
        std::array<cl::size_type, 3> dims = { params.width, params.height, 1 };
        err |= cmdQueue.enqueueReadImage(frontBuffer, CL_TRUE, orig, dims, 0, 0, dataFloats.get());
    }

    err |= cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    err |= cmdQueue.finish();
    verify("Failed to copy pixel buffer to host!");

    // Convert floats to bytes
    int counter = 0;
    for (int i = 0; i < numFloats; i += 4)
    {
        float r = dataFloats[i + 0];
        float g = dataFloats[i + 1];
        float b = dataFloats[i + 2];
        float a = dataFloats[i + 3];

        auto clamp = [](float value) { return std::max(0.0f, std::min(1.0f, value)); };
        dataBytes[counter++] = (unsigned char)(255 * clamp(r / a));
        dataBytes[counter++] = (unsigned char)(255 * clamp(g / a));
        dataBytes[counter++] = (unsigned char)(255 * clamp(b / a));
    }

    // Save image
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(params.width, params.height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, dataBytes.get());
	ilSaveImage(filename.c_str());
    ilDeleteImage(imageID);

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
    size_t s_bytes = sizeof(test_spheres);

    // READ_WRITE due to Apple's OpenCL bug...?
    sphereBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, s_bytes, NULL, &err);
    verify("Sphere buffer creation failed!");

    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, s_bytes, test_spheres);
    verify("Sphere buffer writing failed!");

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
    std::cout << "RenderParam allocation succeeded!" << std::endl;
}

void CLContext::setupStats()
{
    renderStats = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(RenderStats) * 1, NULL, &err);
    verify("RenderStats creation failed!");
    std::cout << "RenderStats allocation succeeded!" << std::endl;

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

const RenderStats CLContext::getStats()
{
    return statsAsync;
}

void CLContext::updateParams(const RenderParams &params)
{
    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(renderParams, CL_TRUE, 0, sizeof(RenderParams), &params);
    verify("RenderParam writing failed");
}

void CLContext::enqueueResetKernel(const RenderParams &params)
{
	std::vector<cl::Memory> pixelBuffer { sharedMemory.back() };
	err = 0;
	err |= cmdQueue.enqueueAcquireGLObjects(&pixelBuffer); // Take hold of PBO
	err |= cmdQueue.enqueueNDRangeKernel(mk_reset, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
	err |= cmdQueue.enqueueReleaseGLObjects(&pixelBuffer);
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
    err |= mk_splat.setArg(4, iteration);
    verify("Failed to set mk_splat arguments!");

    // Splat kernel must be aware of GL-CL sharing
	std::vector<cl::Memory> pixelBuffer { sharedMemory.back() };
    err = cmdQueue.enqueueAcquireGLObjects(&pixelBuffer); // Take hold of PBO
    verify("Failed to enqueue GL object acquisition!");

    // TODO: find out why my GTX 780 won't enqueue 1D kernels! (due to image2d_type?)
    // TODO: also, look at having global wg be a multiple of local wg (or a multiple of 32/64)
    err = cmdQueue.enqueueNDRangeKernel(mk_splat, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&pixelBuffer);
    verify("Failed to enqueue GL object release!");
}

void CLContext::enqueueSplatPreviewKernel(const RenderParams &params)
{
    std::vector<cl::Memory> pixelBuffer{ sharedMemory.back() };
    err = cmdQueue.enqueueAcquireGLObjects(&pixelBuffer); // Take hold of PBO
    verify("Failed to enqueue GL object acquisition!");

    err = cmdQueue.enqueueNDRangeKernel(mk_splat_preview, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat preview kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&pixelBuffer);
    verify("Failed to enqueue GL object release!");
}

void CLContext::enqueueMegaKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration)
{
    err = 0;
    err |= kernel_monolith.setArg(0, sharedMemory[1 - frontBuffer]); // src
    err |= kernel_monolith.setArg(1, sharedMemory[frontBuffer]); // dst
    err |= kernel_monolith.setArg(15, iteration);
    verify("Failed to set megakernel arguments!");

    size_t max_wg_size;
    //err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_gw_size); //CL_KERNEL_WORK_GROUP_SIZE
    max_wg_size = kernel_monolith.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &err);
    verify("Failed to retrieve kernel work group info!");

    ndRangeSizes[0] = 32; //TODO: 32 might be too large
    ndRangeSizes[1] = max_wg_size / ndRangeSizes[0];

    // Multiples of 32
    int wgMultipleWidth = ((params.width & 0x1F) == 0) ? params.width : ((params.width & 0xFFFFFFE0) + 0x20);
    int wgMutipleHeight = (int)(ceil(params.height / (float) ndRangeSizes[1]) * ndRangeSizes[1]);

    cl::NDRange global(wgMultipleWidth, wgMutipleHeight);
    cl::NDRange local(ndRangeSizes[0], ndRangeSizes[1]);

    // Enqueue commands to be executed in order (glFinish called in tracer.cpp)
    err = cmdQueue.enqueueAcquireGLObjects(&sharedMemory); // Take hold of texture
    verify("Failed to enqueue GL object acquisition!");

    #ifdef CPU_DEBUGGING
        err = cmdQueue.enqueueNDRangeKernel(
            kernel_monolith,
            cl::NullRange,                 // offset
            cl::NDRange(params.width, 5),  // global
            cl::NullRange                  // local
        );
    #else
        // Manually select local workgroup size
        //err = cmdQueue.enqueueNDRangeKernel(kernel_monolith, cl::NullRange, global, local);
        
        // Let OpenCL implementation choose optimal local workgroup size
        err = cmdQueue.enqueueNDRangeKernel(kernel_monolith, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    #endif
    verify("Failed to enqueue megakernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    verify("Failed to enqueue GL object release!");
}

void CLContext::finishQueue()
{
    err = cmdQueue.finish();
    verify("Failed to finish command queue!");
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



