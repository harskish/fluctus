#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cl2.hpp"
#include "window.hpp"
#include "tracer.hpp"
#include "math/float3.hpp"
#include "IL/il.h"
#include "IL/ilu.h"

int main(int argc, char* argv[])
{
    Settings &s = Settings::getInstance();

    // Initial size of window
    int width = (argc > 1) ? atoi(argv[1]) : s.getWindowWidth();
    int height = (argc > 2) ? atoi(argv[2]) : s.getWindowHeight();

    ilInit();
    iluInit();
    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    Tracer tracer(width, height);

    // Main loop
    while(tracer.running())
    {
        tracer.update();
    }

    glfwTerminate(); // in tracer destructor?

    return 0;
}

