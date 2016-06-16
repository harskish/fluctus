#include "kernelreader.hpp"

void kernelFromFile(const std::string filename, cl::Context &context, cl::Program &program, int &err)
{
    std::ifstream f(filename);
    std::stringstream buffer;
    buffer << f.rdbuf();
    
    const std::string &tmp = buffer.str();
    program = cl::Program(context, tmp, false, &err);
}