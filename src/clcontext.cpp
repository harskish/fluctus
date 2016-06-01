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

CLContext::CLContext(int gpu, GLuint gl_tex)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    cl::Platform platform = platforms[0];
    std::cout << "Using platform 0" << std::endl;

    platform.getDevices(CL_DEVICE_TYPE_GPU, &clDevices); //CL_DEVICE_TYPE_ALL?
    std::cout << "Forcing GPU device" << std::endl;

    // Init shared context
    #ifdef __APPLE__
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties)kCGLShareGroup, 0
        };
        //context = clCreateContext(props, 1, &(device_id), NULL, NULL, &err); // 1 = number of devices
        context = cl::Context(CL_DEVICE_TYPE_GPU, props, NULL, NULL, &err);
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
    err = program.build(clDevices);
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to build compute program!" << std::endl;
        std::cout << "Build log: " << buildLog << std::endl;
        exit(1);
    }

    // Creating compute kernel from program
    raytracer_kernel = cl::Kernel(program, "trace", &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create compute kernel!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Create OpenCL texture from OpenGL texture
    createCLTexture(gl_tex);
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;

    // TODO: CPP-destructors????

    // Shutdown and cleanup
    /*clReleaseMemObject(pixels);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);*/
}

void CLContext::createCLTexture(GLuint gl_tex) {
    if(pixels) {
        std::cout << "Removing old CL-texture" << std::endl;
        clReleaseMemObject(pixels);
    }

    // CL_MEM_WRITE_ONLY is faster, but we need accumulation...
    pixels = clCreateFromGLTexture(context(), CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, gl_tex, NULL);

    if(!pixels)
        std::cout << "Error: CL-texture creation failed!" << std::endl;
    else
        std::cout << "Created CL-texture at " << pixels << std::endl;
}

// Execute the kernel over the entire range of our 1d input data set
// using the maximum number of work group items for this device
void CLContext::executeKernel()
{
    // Take hold of texture
    std::cout << "Acquiring GL object" << std::endl;
    glFinish();
    
    clEnqueueAcquireGLObjects(cmdQueue(), 1, &pixels, 0, 0, 0);
    

    err = raytracer_kernel.setArg(0, sizeof(cl_mem), &pixels);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to set kernel arguments! " << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &local);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }
    

    int width = 800;
    int height = 600;

    ndRangeSizes[0] = 32; //TODO: 32 might be too large
    ndRangeSizes[1] = local / ndRangeSizes[0];

    std::cout << "Executing kernel..." << std::endl;

    int wgMultipleWidth = ((width & 0x1F) == 0) ? width : ((width & 0xFFFFFFE0) + 0x20);
    int wgMutipleHeight = (int) ceil(height / (float) ndRangeSizes[1]) * ndRangeSizes[1];
    
    cmdQueue.enqueueNDRangeKernel(raytracer_kernel, cl::NullRange,
                cl::NDRange(wgMultipleWidth, wgMutipleHeight),
                cl::NDRange(ndRangeSizes[0], ndRangeSizes[1]));
    
    cmdQueue.finish();
    std::cout << "Kernel execution finished" << std::endl;

    // Release texture for OpenGL to draw it
    std::cout << "Releasing GL object" << std::endl;
    clEnqueueReleaseGLObjects(cmdQueue(), 1, &pixels, 0, 0, NULL);
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



