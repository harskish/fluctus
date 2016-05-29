#include <cmath>
#include "clcontext.hpp"

CLContext::CLContext(int gpu, GLuint gl_tex)
{
    // Connect to a compute device
    std::cout << "Compute device: " << (gpu ? "GPU" : "CPU") << std::endl;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create a device group!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Init shared context
    #ifdef __APPLE__
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties)kCGLShareGroup, 0
        };
        context = clCreateContext(props, 1, &device_id, NULL, NULL, &err); // 1 = number of devices
        if(!context)
        {
            std::cout << "Error: Failed to create shared context" << std::endl;
            std::cout << errorString() << std::endl;
            exit(1);
        }
    #else
        // Only MacOS support for now
        Not yet implemented!
    #endif

    // Create command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if(!commands)
    {
        std::cout << "Error: Failed to create command queue!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Read kenel source from file
    kernelFromFile("src/kernel.cl", context, program, err);
    if (!program)
    {
        std::cout << "Error: Failed to create compute program!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        std::cout << "Error: Failed to build program executable!" << std::endl;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << buffer << std::endl;
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "trace", &err);
    if (!kernel || err != CL_SUCCESS)
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

    // Shutdown and cleanup
    clReleaseMemObject(pixels);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

void CLContext::createCLTexture(GLuint gl_tex) {
    if(pixels) {
        std::cout << "Removing old CL-texture" << std::endl;
        clReleaseMemObject(pixels);
    }

    // CL_MEM_WRITE_ONLY is faster, but we need accumulation...
    pixels = clCreateFromGLTexture(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, gl_tex, NULL);

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
    clEnqueueAcquireGLObjects(commands, 1, &pixels, 0, 0, NULL);

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixels);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to set kernel arguments! " << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    unsigned int count = 800 * 600;
    size_t numLocalGroups = std::ceil(count/local);
    size_t global = local * numLocalGroups;

    std::cout << "Executing kernel..." << std::endl;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        std::cout << "Error: Failed to execute kernel!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    clFinish(commands);
    std::cout << "Kernel execution finished" << std::endl;

    // Release texture for OpenGL to draw it
    std::cout << "Releasing GL object" << std::endl;
    clEnqueueReleaseGLObjects(commands, 1, &pixels, 0, 0, NULL);
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


