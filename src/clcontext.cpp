#include "clcontext.hpp"

CLContext::CLContext(bool gpu)
{
    std::cout << "Compute device: " << (gpu ? "GPU" : "CPU") << std::endl;

    // Fill our data set with random float values
    unsigned int count = DATA_SIZE;
    for(int i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    
    // Connect to a compute device
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create a device group!" << std::endl;
        exit(1);
    }
  
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        std::cout << "Error: Failed to create a compute context!" << std::endl;
        exit(1);
    }

    // Create command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        std::cout << "Error: Failed to create a command commands!" << std::endl;
        exit(1);
    }

    // Read kenel source from file
    kernelFromFile("src/kernel.cl", context, program, err);
    if (!program)
    {
        std::cout << "Error: Failed to create compute program!" << std::endl;
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
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to create compute kernel!" << std::endl;
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        std::cout << "Error: Failed to allocate device memory!" << std::endl;
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to write to source array!" << std::endl;
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to set kernel arguments! " << err << std::endl;
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
        exit(1);
    }
}

CLContext::~CLContext()
{
    std::cout << "Calling CLContext destructor!" << std::endl;

    // Shutdown and cleanup
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

// Execute the kernel over the entire range of our 1d input data set
// using the maximum number of work group items for this device
void CLContext::executeKernel()
{
    unsigned int count = DATA_SIZE;
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        std::cout << "Error: Failed to execute kernel!" << std::endl;
        exit(1);
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        std::cout << "Error: Failed to read output array! " << err << std::endl;
        exit(1);
    }
    
    // Validate our results
    correct = 0;
    for(int i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    std::cout << "Computed '" << correct << "/" << count << "' correct values!" << std::endl;
}



