#include "clcontext.hpp"
#include <stdlib.h>

#ifdef _DEBUG
#define CPU_DEBUGGING
#endif

/* UTILS */

cl::Platform &CLContext::getPlatformByName(std::vector<cl::Platform> &platforms, std::string name) {
    for(cl::Platform &p: platforms) {
        std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
        if(platformName.find(name) != std::string::npos) {
            return p;
        }
    }

    std::cout << "No platform name containing \"" << name << "\" found!" << std::endl;
    return platforms[0];
}

cl::Device &CLContext::getDeviceByName(std::vector<cl::Device> &devices, std::string name) {
    for(cl::Device &d: devices) {
        std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
        if(deviceName.find(name) != std::string::npos) {
            return d;
        }
    }

    std::cout << "No device name containing \"" << name << "\" in selected context!" << std::endl;
    return devices[0];
}

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


/* CLASS METHODS */

CLContext::CLContext(GLuint *textures)
{
    // Remove kernel caching to always get build logs (on NVIDIA hardware)
    #ifdef _WIN32
        _putenv_s("CUDA_CACHE_DISABLE", "1");
    #endif

    //printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    platform = getPlatformByName(platforms, Settings::getInstance().getPlatformName());
    std::cout << "PLATFORM: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);

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

    // Create placeholder environment map
    float rgb[3] = { 0.0f, 0.0f, 0.0f };
    createEnvMap(rgb, 1, 1);

    // Setup RenderParams
    setupParams();

    // Create OpenCL buffer from OpenGL PBO
    createTextures(textures);

    // Allocate device memory for scene
    setupScene();
    
    // Build kernels, set their params
    setupKernels();
}

void CLContext::setupKernels()
{
    initMCBuffers();
    setupRayGenKernel();
    setupNextVertexKernel();
    setupSplatKernel();
    setupMegaKernel();
}

// Init state buffers (rays, tasks) needed by microkernels
void CLContext::initMCBuffers()
{
    const size_t t_bytes = NUM_TASKS * sizeof(GPUTaskState);
    const size_t r_bytes = NUM_TASKS * sizeof(Ray);

    raysBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, r_bytes, NULL, &err);
    verify("Ray buffer creation failed!");

    tasksBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, t_bytes, NULL, &err);
    verify("Task buffer creation failed!");

    // Init tasks buffer
    GPUTaskState *initialTaskStates = new GPUTaskState[NUM_TASKS];
    std::for_each(initialTaskStates + 0, initialTaskStates + NUM_TASKS, [](GPUTaskState &s) { s.phase = MK_GENERATE_CAMERA_RAY; s.pdf = 1.0f; s.T = float3(1.0f); s.seed = (unsigned int)rand(); });
    err = cmdQueue.enqueueWriteBuffer(tasksBuffer, CL_TRUE, 0, t_bytes, initialTaskStates);
    delete[] initialTaskStates;
    verify("Task buffer writing failed!");

    const size_t memoryUsageMiB = (t_bytes + r_bytes) / (2 << 19);
    std::cout << "Microkernel state data: " << memoryUsageMiB << " MiB" << std::endl;
}

// Build kernel based on file name and entrypoint method name. Save compiled kernel in target.
// Platform and build specific build options are automatically set.
void CLContext::buildKernel(cl::Kernel &target, std::string fileName, std::string methodName)
{
    // Kernel already exists
    if (target()) return;

    // Read kernel source from file
    std::string kernelPath = "src/" + fileName;

    cl::Program program;
    kernelFromFile(kernelPath, context, program, err);
    verify("Failed to create compute program for " + fileName);

    // Build kernel source (create compute program)
    // Define "GPU" to disable cl-prefixed types in shared headers (cl_float4 => float4 etc.)
    std::string buildOpts = "-DGPU -I./src";

    if (platformIsNvidia(platform)) buildOpts += " -cl-nv-verbose";

    #ifdef CPU_DEBUGGING
        buildOpts += " -g -s \"" + getAbsolutePath(kernelPath) + "\"";
    #endif

    err = program.build(clDevices, buildOpts.c_str());
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

    // Check build log
    std::cout << "[" << fileName << " build log]\n" << buildLog << std::endl;
    verify("Failed to build compute program!");

    // Creating compute kernel from program
    target = cl::Kernel(program, methodName.c_str(), &err);
    verify("Failed to create compute kernel!");
}

void CLContext::setupMegaKernel()
{
    buildKernel(kernel_monolith, "kernel_monolith.cl", "trace");

    int i = 0;
    err = 0;
    err |= kernel_monolith.setArg(i++, sharedMemory[1]); // src
    err |= kernel_monolith.setArg(i++, sharedMemory[0]); // dst
    err |= kernel_monolith.setArg(i++, sphereBuffer);
    err |= kernel_monolith.setArg(i++, lightBuffer);
    err |= kernel_monolith.setArg(i++, triangleBuffer);
    err |= kernel_monolith.setArg(i++, materialBuffer);
    err |= kernel_monolith.setArg(i++, nodeBuffer);
    err |= kernel_monolith.setArg(i++, indexBuffer);
    err |= kernel_monolith.setArg(i++, environmentMap);
    err |= kernel_monolith.setArg(i++, renderParams);
    err |= kernel_monolith.setArg(i++, 0); // iteration
    verify("Failed to set kernel arguments!");
}

void CLContext::setupRayGenKernel()
{
    buildKernel(mk_raygen, "mk_raygen.cl", "genCameraRays");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_raygen.setArg(i++, raysBuffer);
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
    err |= mk_next_vertex.setArg(i++, raysBuffer);
    err |= mk_next_vertex.setArg(i++, tasksBuffer);
    err |= mk_next_vertex.setArg(i++, materialBuffer);
    err |= mk_next_vertex.setArg(i++, triangleBuffer);
    err |= mk_next_vertex.setArg(i++, nodeBuffer);
    err |= mk_next_vertex.setArg(i++, indexBuffer);
    err |= mk_next_vertex.setArg(i++, renderParams);
    err |= mk_next_vertex.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_next_vertex arguments!");
}

void CLContext::setupSplatKernel()
{
    buildKernel(mk_splat, "mk_splat.cl", "splat");

    // Set initial kernel params
    int i = 0;
    err = 0;
    err |= mk_splat.setArg(i++, raysBuffer);
    err |= mk_splat.setArg(i++, tasksBuffer);
    // The front/back buffers change every iteration
    err |= mk_splat.setArg(i++, sharedMemory[1]); // src
    err |= mk_splat.setArg(i++, sharedMemory[0]); // dst
    err |= mk_splat.setArg(i++, renderParams);
    err |= mk_splat.setArg(i++, NUM_TASKS);
    verify("Failed to set mk_splat arguments!");
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;
}

void CLContext::createTextures(GLuint *tex_arr)
{
    if (sharedMemory.size() > 0)
    {
        std::cout << "Removing old textures" << std::endl;
        sharedMemory.clear(); // memory freed by cl-cpp-wrapper
    }

    sharedMemory.push_back(cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex_arr[0], &err));
    sharedMemory.push_back(cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex_arr[1], &err));

    verify("CL-texture creation failed!");
}

void CLContext::createEnvMap(float *data, int width, int height)
{
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

    const cl::ImageFormat format(CL_RGBA, CL_FLOAT);
    environmentMap = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width, height, 0, rgba, &err);
    verify("Environment map creation failed!");

    delete [] rgba;
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
    renderParams = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(RenderParams), NULL, &err);
    verify("Params buffer creation failed!");

    std::cout << "RenderParam allocation succeeded!" << std::endl;
}

void CLContext::updateParams(const RenderParams &params)
{
    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(renderParams, CL_TRUE, 0, sizeof(RenderParams), &params);
    verify("RenderParam writing failed");

    // std::cout << "RenderParams updated!" << std::endl;
}

void CLContext::executeRayGenKernel(const RenderParams &params)
{
    // Set updated RenderParams
    err = mk_raygen.setArg(2, renderParams);
    verify("Failed to set mk_raygen RenderParams!");

    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_raygen, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue kernel!");
}

void CLContext::executeNextVertexKernel(const RenderParams &params)
{
    // Set updated RenderParams
    err = mk_next_vertex.setArg(6, renderParams);
    verify("Failed to set mk_next_vertex RenderParams!");

    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_next_vertex, cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue kernel!");
}

void CLContext::executeSplatKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration)
{
    err = 0;
    err |= mk_splat.setArg(2, sharedMemory[1 - frontBuffer]); // src
    err |= mk_splat.setArg(3, sharedMemory[frontBuffer]); // dst
    err |= mk_splat.setArg(4, renderParams);
    verify("Failed to set mk_splat arguments!");

    // Splat kernel must be aware of GL-CL sharing
    err = cmdQueue.enqueueAcquireGLObjects(&sharedMemory); // Take hold of texture
    verify("Failed to enqueue GL object acquisition!");

    // TODO: find out why my GTX 780 won't enqueue 1D kernels!
    // TODO: also, look at having global wg be a multiple of local wg (or a multiple of 32/64)
    err = cmdQueue.enqueueNDRangeKernel(mk_splat, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    verify("Failed to enqueue GL object release!");

    err = cmdQueue.finish();
    verify("Failed to finish command queue!");
}

void CLContext::executeMegaKernel(const RenderParams &params, const int frontBuffer, const cl_uint iteration)
{
    err = 0;
    err |= kernel_monolith.setArg(0, sharedMemory[1 - frontBuffer]); // src
    err |= kernel_monolith.setArg(1, sharedMemory[frontBuffer]); // dst
    err |= kernel_monolith.setArg(9, renderParams);
    err |= kernel_monolith.setArg(10, iteration);
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
    verify("Failed to enqueue kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    verify("Failed to enqueue GL object release!");
    
    err = cmdQueue.finish();
    verify("Failed to finish command queue!");
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

void CLContext::verify(std::string msg)
{
    if(err != CL_SUCCESS)
    {
        std::string message = msg + " (" + errorString() + ")";
        std::cout << message << std::endl;
#ifdef WIN32
        system("pause");
#endif
        exit(1);
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



