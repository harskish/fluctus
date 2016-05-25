#include "kernelreader.hpp"

void kernelFromFile(const std::string filename, cl_context &context, cl_program &program, int &err) {
    std::ifstream f(filename);
    std::stringstream buffer;
    buffer << f.rdbuf();
    
    const std::string &tmp = buffer.str();
    const char *cstr = tmp.c_str();
    size_t programSize = tmp.size();
    
    program = clCreateProgramWithSource(context, 1,
            (const char**) &cstr, &programSize, &err);
}