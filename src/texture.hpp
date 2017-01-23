#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <string>
#include <iostream>
#include "cl2.hpp"
#include "IL/il.h"
#include "IL/ilu.h"

/* Reads a texture using DevIL */

class Texture
{
public:
    //Texture() : width(0), height(0), data(NULL) {} // default constructor
    Texture(const std::string path, const std::string name);
    ~Texture() { if (data) delete[] data; }

    cl_uchar *getData() { return data; }
    cl_uint getWidth() { return width; }
    cl_uint getHeight() { return height; }
    std::string getName() { return name; }

private:
    std::string name; // used to check if a specific texture is already loaded
    cl_uint width, height;
    cl_uchar *data; // eventually passed to OpenCL
};