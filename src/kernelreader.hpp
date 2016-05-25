#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>

void kernelFromFile(const std::string filename, cl_context &context, cl_program &program, int &err);