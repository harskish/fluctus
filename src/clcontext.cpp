#include "clcontext.hpp"

#ifdef _DEBUG
#define CPU_DEBUGGING
#endif

CLContext::CLContext(GLuint gl_PBO)
{
    //printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

	#ifdef CPU_DEBUGGING
		cl::Platform platform = platforms[2];
	#else
		cl::Platform platform = platforms[1];
	#endif

	std::cout << "PLATFORM: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);

    // Macbook pro 15 fix
    #ifdef __APPLE__
    clDevices.erase(clDevices.begin());
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
    if(err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create shared context" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }
    device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
    std::cout << "Using device nr. 0 of context" << std::endl;

    // Create command queue for context
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
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create compute kernel!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

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

    if(err != CL_SUCCESS)
    {
        std::cout << "Error: CL-PBO creation failed!" << std::endl;
        exit(1);
    } else {
        std::cout << "Created CL-PBO at " << (sharedMemory.back())() << std::endl;
    }
}

void CLContext::setupScene()
{
    size_t s_bytes = sizeof(test_spheres);

    // READ_WRITE due to Apple's OpenCL bug...?
    sphereBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, s_bytes, NULL, &err);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: sphere buffer creation failed!" << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, s_bytes, test_spheres);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: scene buffer writing failed!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // Lights
    size_t l_bytes = sizeof(test_lights);
    lightBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, l_bytes, NULL, &err);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: light buffer creation failed!" << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    err = cmdQueue.enqueueWriteBuffer(lightBuffer, CL_TRUE, 0, l_bytes, test_lights);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: light buffer writing failed!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

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

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: test buffer creation failed!" << err << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    std::cout << "RenderParam allocation succeeded!" << std::endl;
}

void CLContext::updateParams(const RenderParams &params)
{
    // Blocking write!
    err = cmdQueue.enqueueWriteBuffer(renderParams, CL_TRUE, 0, sizeof(RenderParams), &params);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error: RenderParam writing failed!" << std::endl;
        std::cout << errorString() << std::endl;
        exit(1);
    }

    // std::cout << "RenderParams updated!" << std::endl;
}

void CLContext::executeKernel(const RenderParams &params)
{
    int i = 0;
    err = 0;
    err |= pt_kernel.setArg(i++, sharedMemory.back()); // output buffer
    err |= pt_kernel.setArg(i++, sphereBuffer);
    err |= pt_kernel.setArg(i++, lightBuffer);
    err |= pt_kernel.setArg(i++, triangleBuffer);
    err |= pt_kernel.setArg(i++, nodeBuffer);
    err |= pt_kernel.setArg(i++, indexBuffer);
    err |= pt_kernel.setArg(i++, renderParams);

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

    // Multiples of 32
    int wgMultipleWidth = ((params.width & 0x1F) == 0) ? params.width : ((params.width & 0xFFFFFFE0) + 0x20);
    int wgMutipleHeight = (int)(ceil(params.height / (float) ndRangeSizes[1]) * ndRangeSizes[1]);

    cl::NDRange global(wgMultipleWidth, wgMutipleHeight);
    cl::NDRange local(ndRangeSizes[0], ndRangeSizes[1]);

    // Enqueue commands to be executed in order
    glFinish();
    cmdQueue.enqueueAcquireGLObjects(&sharedMemory); // Take hold of texture

	#ifdef CPU_DEBUGGING
		cmdQueue.enqueueNDRangeKernel(
			pt_kernel,
			cl::NullRange,                 // offset
			cl::NDRange(params.width, 5),  // global
			cl::NullRange                  // local
		);
	#else
		cmdQueue.enqueueNDRangeKernel(pt_kernel, cl::NullRange, global, local);
	#endif

    cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    cmdQueue.finish();
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
        std::cout << msg << std::endl << errorString() << std::endl;
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



