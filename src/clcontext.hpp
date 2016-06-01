#pragma once

#define DATA_SIZE (2048)
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/OpenGL.h>
#endif

#include <GLFW/glfw3.h> // texture conversion stuff
#include <iostream>
#include <string>
#include "cl2.hpp"
#include "kernelreader.hpp"

class CLContext
{
public:
    CLContext(int gpu, GLuint gl_tex);
    ~CLContext();

    void executeKernel();
    void createCLTexture(GLuint gl_tex);
private:
    void printDevices();
    std::string errorString();

    int err;                            // error code returned from api calls

    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_platform_id platform;
    
    cl_mem pixels = 0;                  // device memory used for pixel data
};