#include "Kernel.hpp"
#include "kernelreader.hpp"
#include <iostream>
#include <cassert>
#include "utils.h"

//#ifdef _DEBUG
//#define CPU_DEBUGGING
//#endif

FLT_NAMESPACE_BEGIN

std::string Kernel::globalBuildOpts = "";
void* Kernel::userPtr = nullptr;

// Check CL command success
void Kernel::verify(int code, const std::string msg) {
    if (code != CL_SUCCESS)
    {
        std::string message = msg + " (" + getCLErrorString(code) + ")";
        std::cout << message << std::endl;
        waitExit();
    }
}

void Kernel::build(std::string path, std::string entryPoint, cl::Context& context, cl::Device& device, cl::Platform& platform, bool setArgs)
{
    // No need to recompile, just update arguments
    if (m_kernel() && !configHasChanged())
    {
        if (setArgs)
            this->setArgs();
        return;
    }

    const std::string filename = getFileName(path);

    this->context = &context;
    this->device = &device;
    this->platform = &platform;
    this->srcPath = path;
    this->entryPoint = entryPoint;

    if (m_kernel())
        std::cout << "Rebuilding kernel " << filename << std::endl;

    // Define build options based on global + specialized options
    std::string buildOpts = globalBuildOpts + getAdditionalBuildOptions();
#ifdef CPU_DEBUGGING
    buildOpts += " -g -s \"" + getAbsolutePath(srcPath) + "\"";
#endif
    this->lastBuildOpts = buildOpts;
    cl::Program program;

    // CPU debugging segfaults if trying to use cached kernel!
    // Also need to let the driver do the include handling
    int err = 0;
#ifdef CPU_DEBUGGING
    kernelFromSource(srcPath, context, program, err);
    cl::vector<cl::Device> devices = { device };
    err = program.build(devices, buildOpts.c_str());

    // Check build log
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    if (buildLog.length() > 2)
        std::cout << "\n[" << path << " build log]:" << buildLog << std::endl;

    verify(err, "Kernel compilation failed");
#else
    // Build program using cache or sources
    program = kernelFromFile(path, buildOpts, platform, context, device, err);
    verify(err, "Failed to create kernel program");
#endif

    // Creating compute kernel from program
    m_kernel = cl::Kernel(program, entryPoint.c_str(), &err);
    verify(err, "Failed to create compute kernel!");

    // Get kernel argument names
    // NB: kernels built from binaries SHOULD NOT have arg info, but they do at least on Intel/NV!
    argMap.clear();
    cl_uint numArgs = m_kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
    verify(err, "Getting KERNEL_NUM_ARGS failed for " + filename);
    for (cl_uint i = 0; i < numArgs; i++)
    {
        auto argname = m_kernel.getArgInfo<CL_KERNEL_ARG_NAME>(i, &err);
        verify(err, "Getting CL_KERNEL_ARG_NAME failed for " + filename);
        argMap[argname] = i; // save to mapping
    }

    // Set default arguments
    this->setArgs();
}

void Kernel::rebuild(bool setArgs)
{
    build(srcPath, entryPoint, *context, *device, *platform, setArgs);
}

bool Kernel::configHasChanged()
{
    std::string buildOpts = globalBuildOpts + getAdditionalBuildOptions();
#ifdef CPU_DEBUGGING
    buildOpts += " -g -s \"" + getAbsolutePath(srcPath) + "\"";
#endif
    return (buildOpts.compare(lastBuildOpts) != 0);
}


FLT_NAMESPACE_END
