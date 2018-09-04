#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include "cl2.hpp"
#include "utils.h"

void kernelFromSource(const std::string filename, cl::Context &context, cl::Program &program, int &err);
void kernelFromSourceExpanded(const std::string filename, cl::Context &context, cl::Program &program, int &err);
void kernelFromBinary(const std::string filename, cl::Context &context, cl::Device &device, cl::Program &program, int &err);
cl::Program kernelFromFile(const std::string filename, const std::string buildOpts, cl::Platform &platform, cl::Context &context, cl::Device &device, int &err);

std::string readKernel(std::string path, std::vector<std::string> &incl);
std::string readKernel(std::string path);