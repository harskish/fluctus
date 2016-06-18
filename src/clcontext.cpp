#include <cmath>
#include <vector>
#include "clcontext.hpp"


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

CLContext::CLContext(GLuint gl_PBO)
{
    printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms[0];
    std::cout << "Using platform 0" << std::endl;

    platform.getDevices(CL_DEVICE_TYPE_GPU, &clDevices); //CL_DEVICE_TYPE_ALL?
    std::cout << "Forcing GPU device" << std::endl;

    // Macbook pro 15 fix
    // clDevices.erase(clDevices.begin());

    // Init shared context
    #ifdef __APPLE__
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties)kCGLShareGroup, 0
        };
        context = cl::Context(clDevices, props, NULL, NULL, &err); //CL_DEVICE_TYPE_GPU instead of clDevices?
        if(err != CL_SUCCESS)
        {
            std::cout << "Error: Failed to create shared context" << std::endl;
            std::cout << errorString() << std::endl;
            exit(1);
        }
        device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::cout << "Using device nr. 0 of context" << std::endl;
    #else
        // Only MacOS support for now
        Not yet implemented!
    #endif


    cmdQueue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create command queue!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Read kenel source from file
    cl::Program program;
    kernelFromFile("src/kernel.cl", context, program, err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create compute program! " << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Build kernel source (create compute program)
    // Define "GPU" to disable cl-prefixed types in shared headers (cl_float4 => float4 etc.)
    err = program.build(clDevices, "-I./src -DGPU");
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to build compute program!" << std::endl;
        std::cout << "Build log: " << buildLog << std::endl;
        exit(1);
    }

    // Creating compute kernel from program
    pt_kernel = cl::Kernel(program, "trace", &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create compute kernel!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Create OpenCL buffer from OpenGL PBO
    createPBO(gl_PBO);

    // Allocate device memory for scene
    setupScene();
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;

    // TODO: CPP-destructors????

    /* Shutdown and cleanup
    clReleaseMemObject(pixels);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);*/
}

void CLContext::createPBO(GLuint gl_PBO)
{
    if(cl_PBO) {
        std::cout << "Removing old CL-PBO" << std::endl;
        clReleaseMemObject(cl_PBO);
    }

    // CL_MEM_WRITE_ONLY is faster, but we need accumulation...
    cl_PBO = clCreateFromGLBuffer(context(), CL_MEM_READ_WRITE, gl_PBO, &err);

    if(!cl_PBO)
        std::cout << "Error: CL-PBO creation failed!" << std::endl;
    else
        std::cout << "Created CL-PBO at " << cl_PBO << std::endl;
}

void CLContext::setupScene()
{
    // READ_WRITE due to Apple's OpenCL bug...?
    size_t s_bytes = sizeof(test_spheres);
    std::cout << "cl_float size: " << sizeof(cl_float) << "B" << std::endl;
    std::cout << "cl_float4 size: " << sizeof(cl_float4) << "B" << std::endl;
    std::cout << "Sphere size: " << sizeof(Sphere) << "B" << std::endl;
    std::cout << "Sphere buffer size: " << s_bytes << "B" << std::endl;

    sphereBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, s_bytes, NULL, &err);

    if (err != CL_SUCCESS) {
        std::cout << "Error: scene buffer creation failed!" << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, s_bytes, test_spheres);

    if (err != CL_SUCCESS) {
        std::cout << "Error: scene buffer writing failed!" << std::endl;
        std::cout << errorString() << std::endl;
        std::cout << "Scene buffer is at: " << test_spheres << std::endl;
        exit(1);
    }

    std::cout << "Scene initialization succeeded!" << std::endl;
}

void CLContext::executeKernel(const unsigned int width, const unsigned int height)
{
    static double t0 = glfwGetTime();

    // TODO: don't recreate params every time

    // Render parameters to be passed to kernel
    RenderParams params;
    params.width = width;
    params.height = height;
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);
    params.sin2 = pow(sin(glfwGetTime() - t0), 2);

    Camera cam;
    cam.pos = {{ 0.0f, 0.1f + params.sin2, 2.5f + params.sin2, 0.0f }};
    cam.dir = {{ 0.0f, -0.2f * params.sin2, -1.0f, 0.0f }};
    cam.fov = 70.0f;
    params.camera = cam;

    // Take hold of texture
    glFinish();
    clEnqueueAcquireGLObjects(cmdQueue(), 1, &cl_PBO, 0, 0, 0);

    err = 0;
    err |= pt_kernel.setArg(0, sizeof(cl_mem), &cl_PBO);
    err |= pt_kernel.setArg(1, sphereBuffer);
    err |= pt_kernel.setArg(2, params);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to set kernel arguments! " << err << std::endl;
        exit(1);
    }

    size_t max_gw_size;
    err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_gw_size);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }


    ndRangeSizes[0] = 32; //TODO: 32 might be too large
    ndRangeSizes[1] = max_gw_size / ndRangeSizes[0];

    //std::cout << "Executing kernel..." << std::endl;

    // Multiples of 32
    int wgMultipleWidth = ((width & 0x1F) == 0) ? width : ((width & 0xFFFFFFE0) + 0x20);
    int wgMutipleHeight = (int) ceil(height / (float) ndRangeSizes[1]) * ndRangeSizes[1];

    cl::NDRange global(wgMultipleWidth, wgMutipleHeight);
    cl::NDRange local(ndRangeSizes[0], ndRangeSizes[1]);

    cmdQueue.enqueueNDRangeKernel(pt_kernel, cl::NullRange, global, local);

    cmdQueue.finish();
    //std::cout << "Kernel execution finished" << std::endl;

    // Release texture for OpenGL to draw it
    //std::cout << "Releasing GL object" << std::endl;
    clEnqueueReleaseGLObjects(cmdQueue(), 1, &cl_PBO, 0, 0, NULL);
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



