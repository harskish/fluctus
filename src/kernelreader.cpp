#include "kernelreader.hpp"

void kernelFromFile(const std::string filename, cl::Context &context, cl::Program &program, int &err)
{
    std::ifstream f(filename);
    std::stringstream buffer;
    buffer << f.rdbuf();
    
    const std::string &tmp = buffer.str();
    //const char *cstr = tmp.c_str();
    //size_t programSize = tmp.size();

    //cl::Program::Sources cl_source(1, std::make_pair(cstr, programSize + 1));
    //cl::Program::Sources cl_source;
    //cl_source.push_back(tmp);
    program = cl::Program(context, tmp, false, &err);
}