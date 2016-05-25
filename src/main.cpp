#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <GLFW/glfw3.h>
#include "cl2.hpp"
#include "window.hpp"
#include "clcontext.hpp"

int main(int argc, char* argv[])
{
    const int USE_GPU = (argc > 1) ? atoi(argv[1]) : 1;

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    Window window(600, 400);
    CLContext ctx(USE_GPU);
    ctx.executeKernel();

    // Main loop
    while(window.available())
    {
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}

