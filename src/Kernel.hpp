#pragma once

#define FLT_NAMESPACE_BEGIN namespace flt {
#define FLT_NAMESPACE_END }

// OCL 1.2+ needed for clGetKernelArgInfo
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"
#include <string>
#include <iostream>
#include <map>

FLT_NAMESPACE_BEGIN

class Kernel
{
public:
    Kernel(void) = default;
    ~Kernel(void) = default;

    explicit operator bool() const { return m_kernel() != nullptr; }
    operator cl::Kernel&() { return m_kernel; }

    // Build, but only if needed!
    void build(std::string path, std::string entryPoint, cl::Context& context, cl::Device& device, cl::Platform& platform, bool setArgs = true);
    void rebuild(bool setArgs);

    //cl::Kernel& getKernel() { return m_kernel; }

    template <typename T>
    cl_int setArg(const std::string name, const T& value) {
        auto it = argMap.find(name);
        if (it == argMap.end())
        {
            std::cout << "Kernel " << srcPath << " has no argument '" << name << "'" << std::endl;
            throw std::runtime_error("Unknown kernel argument " + name);
        }
        else
        {
            return m_kernel.setArg(it->second, value);
        }
    }

    bool hasArg(const std::string name) { return argMap.find(name) != argMap.end(); }

    // For accessing compilation settings and device buffers
    static void setUserPointer(void* p) { Kernel::userPtr = p; }
    static void setBuildOptions(std::string s) { globalBuildOpts = s; }

private:
    // For checking if recompilation is necessary
    bool configHasChanged();
    
    // Cached for recompilation
    cl::Context* context;
    cl::Device* device;
    cl::Platform* platform;

    static std::string globalBuildOpts;
    std::string srcPath;
    std::string entryPoint;
    cl::Kernel m_kernel;
    std::string lastBuildOpts; // for detecting need to recompile
    std::map<std::string, cl_uint> argMap;

protected:
    virtual std::string getAdditionalBuildOptions() { return ""; };
    virtual void setArgs() = 0;

    // Check CL return code
    void verify(int code, const std::string msg);

    static void* userPtr;
};

FLT_NAMESPACE_END
