#include "clcontext.hpp"

#ifdef _DEBUG
#define CPU_DEBUGGING
#endif

cl::Platform &CLContext::getPlatformByName(std::vector<cl::Platform> &platforms, std::string name) {
    for(cl::Platform &p: platforms) {
        std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
        if(platformName == name) {
            return p;
        }
    }

    std::cout << "No platform with name \"" << name << "\" found, using default" << std::endl;
    return platforms[0];
}

cl::Device &CLContext::getDeviceByName(cl::Context &context, std::string name) {
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    for(cl::Device &d: devices) {
        std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
        if(deviceName == name) {
            return d;
        }
    }

    std::cout << "No device with name \"" << name << "\" in selected context, using default" << std::endl;
    return devices[0];
}

CLContext::CLContext(GLuint gl_PBO)
{
    //printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = getPlatformByName(platforms, Settings::getInstance().getPlatformName());
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
    device = getDeviceByName(context, Settings::getInstance().getDeviceName());
    std::cout << "DEVICE: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create command queue for context
    cmdQueue = cl::CommandQueue(context, device, 0, &err);
    verify("Failed to create command queue!");

    // Read kernel source from file
    cl::Program program;
    kernelFromFile("src/kernel.cl", context, program, err);
    verify("Failed to create compute program!");

    // Build kernel source (create compute program)
    // Define "GPU" to disable cl-prefixed types in shared headers (cl_float4 => float4 etc.)
    #ifdef CPU_DEBUGGING
        err = program.build(clDevices, "-DGPU -g -s \"C:\\Users\\Erik\\code\\cltrace\\src\\kernel.cl\"");
    #else
        err = program.build(clDevices, "-I./src -DGPU");
    #endif
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to build compute program!" << std::endl;
        std::cout << "Build log: " << buildLog << std::endl;
        exit(1);
    }

    // Creating compute kernel from program
    pt_kernel = cl::Kernel(program, "trace", &err);
    verify("Failed to create compute kernel!");

    // Create OpenCL buffer from OpenGL PBO
    createPBO(gl_PBO);

    // Allocate device memory for scene and rendering parameters
    setupScene();
    setupParams();
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;
}

void CLContext::createPBO(GLuint gl_PBO)
{
    if(sharedMemory.size() > 0) {
        std::cout << "Removing old CL-PBO" << std::endl;
        sharedMemory.clear(); // memory freed by cl-cpp-wrapper
    }

    // CL_MEM_WRITE_ONLY is faster, but we need accumulation...
    sharedMemory.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, gl_PBO, &err));

    verify("CL-PBO creation failed!");
    std::cout << "Created CL-PBO at " << (sharedMemory.back())() << std::endl;
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
    sphereBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, s_bytes, NULL, &err);
    verify("Sphere buffer creation failed!");

    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, s_bytes, test_spheres);
    verify("Sphere buffer writing failed!");

    // Lights
    size_t l_bytes = sizeof(test_lights);
    lightBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, l_bytes, NULL, &err);
    verify("Light buffer creation failed!");

    err = cmdQueue.enqueueWriteBuffer(lightBuffer, CL_TRUE, 0, l_bytes, test_lights);
    verify("Light buffer writing failed!");

    std::cout << "Scene initialization succeeded!" << std::endl;
}

// TODO: Check if Apple still breaks when using CL_MEM_READ
void CLContext::createBVHBuffers(std::vector<RTTriangle> *tris, std::vector<cl_uint> *indices, std::vector<Node> *nodes)
{
    size_t t_bytes = tris->size() * sizeof(RTTriangle);
    size_t i_bytes = indices->size() * sizeof(cl_uint);
    size_t n_bytes = nodes->size() * sizeof(Node);

    // Allocate memory for buffers
    triangleBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, t_bytes, NULL, &err);
    verify("Triangle buffer creation failed!");

    indexBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, i_bytes, NULL, &err);
    verify("Index buffer creation failed!");

    nodeBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, n_bytes, NULL, &err);
    verify("Node buffer creation failed!");

    // Write data to buffers
    err = cmdQueue.enqueueWriteBuffer(triangleBuffer, CL_TRUE, 0, t_bytes, tris->data());
    verify("Triangle buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, i_bytes, indices->data());
    verify("Index buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(nodeBuffer, CL_TRUE, 0, n_bytes, nodes->data());
    verify("Node buffer writing failed!");
}

// Passing structs to kernels is broken in several drivers (e.g. GT 750M on MacOS)
// Allocating memory for the rendering params is more compatible
void CLContext::setupParams()
{
    renderParams = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(RenderParams), NULL, &err);
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

void CLContext::executeKernel(const RenderParams &params, const cl_uint iteration)
{
    int i = 0;
    err = 0;
    err |= pt_kernel.setArg(i++, sharedMemory.back()); // output buffer
    err |= pt_kernel.setArg(i++, sphereBuffer);
    err |= pt_kernel.setArg(i++, lightBuffer);
    err |= pt_kernel.setArg(i++, triangleBuffer);
    err |= pt_kernel.setArg(i++, nodeBuffer);
    err |= pt_kernel.setArg(i++, indexBuffer);
    err |= pt_kernel.setArg(i++, environmentMap);
    err |= pt_kernel.setArg(i++, renderParams);
    err |= pt_kernel.setArg(i++, iteration);
    verify("Failed to set kernel arguments!");

    size_t max_gw_size;
    err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_gw_size);
    verify("Failed to retrieve kernel work group info!");

    ndRangeSizes[0] = 32; //TODO: 32 might be too large
    ndRangeSizes[1] = max_gw_size / ndRangeSizes[0];

    // Multiples of 32
    int wgMultipleWidth = ((params.width & 0x1F) == 0) ? params.width : ((params.width & 0xFFFFFFE0) + 0x20);
    int wgMutipleHeight = (int)(ceil(params.height / (float) ndRangeSizes[1]) * ndRangeSizes[1]);

    cl::NDRange global(wgMultipleWidth, wgMutipleHeight);
    cl::NDRange local(ndRangeSizes[0], ndRangeSizes[1]);

    // Enqueue commands to be executed in order
    glFinish();
    err = cmdQueue.enqueueAcquireGLObjects(&sharedMemory); // Take hold of texture
    verify("Failed to enqueue GL object acquisition!");

    #ifdef CPU_DEBUGGING
        err = cmdQueue.enqueueNDRangeKernel(
            pt_kernel,
            cl::NullRange,                 // offset
            cl::NDRange(params.width, 5),  // global
            cl::NullRange                  // local
        );
    #else
        //err = cmdQueue.enqueueNDRangeKernel(pt_kernel, cl::NullRange, global, local);
        err = cmdQueue.enqueueNDRangeKernel(pt_kernel, cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
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



