#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <glad/glad.h>
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

    // Initial size of windowg
    int width = (argc > 1) ? atoi(argv[1]) : s.getWindowWidth();
    int height = (argc > 2) ? atoi(argv[2]) : s.getWindowHeight();

    ilInit();
    iluInit();
    ilEnable(IL_ORIGIN_SET);
    ilEnable(IL_FILE_OVERWRITE);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);

    if (!glfwInit())
    {
        std::cout << "Could not initialize GLFW" << std::endl;
        waitExit();
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

