#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <GLFW/glfw3.h>
#include "cl2.hpp"
#include "window.hpp"
#include "clcontext.hpp"

int main(int argc, char* argv[])
{
    // Initial size of window
    int width = (argc > 1) ? atoi(argv[1]) : 800;
    int height = (argc > 2) ? atoi(argv[2]) : 600;

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    std::cout << "Window dimensions: [" << width << ", " << height << "]" << std::endl;
    Window window(width, height);

    CLContext ctx(window.getPBO());
    window.setCLCtx(&ctx);
    
    //ctx.executeKernel();
    //window.repaint();

    int fbw, fbh;

    // Main loop
    while(window.available())
    {
        // Do stuff
        window.getFBSize(fbw, fbh);
        ctx.executeKernel((unsigned int)fbw, (unsigned int)fbh);
        window.repaint();
    }

    glfwTerminate();

    return 0;
}

