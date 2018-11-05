#include "tracer.hpp"
#include "IL/il.h"
#include "IL/ilu.h"
#include "settings.hpp"
#include "utils.h"

int main(int argc, char* argv[])
{
    Settings &s = Settings::getInstance();

    bool interactiveMode = true;

    // Initial size of windowg
    int width = (argc > 1) ? atoi(argv[1]) : s.getWindowWidth();
    int height = (argc > 2) ? atoi(argv[2]) : s.getWindowHeight();
    int spp = 0;

    if (argc > 3)
    {
        spp = atoi(argv[3]);
        interactiveMode = false;
    }

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

    if (interactiveMode)
        tracer.renderInteractive();
    else
        tracer.renderSingle(spp);

    glfwTerminate();

    return 0;
}

