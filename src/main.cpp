#include "tracer.hpp"
#include "IL/il.h"
#include "IL/ilu.h"
#include "settings.hpp"
#include "utils.h"

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

