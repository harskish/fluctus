#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include "cl2.hpp"

void kernelFromFile(const std::string filename, cl::Context &context, cl::Program &program, int &err);