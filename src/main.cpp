#include "tracer.hpp"
#include "IL/il.h"
#include "IL/ilu.h"
#include "settings.hpp"
#include "utils.h"
#include <string>
#include <vector>
#include <tclap/CmdLine.h>

int main(int argc, char* argv[])
{
    Settings &s = Settings::getInstance();

    int width;
    int height;
    int spp;
    bool interactiveMode;
    std::vector<std::string> scenes;

    // Parse command line arguments
    try
    {
        TCLAP::CmdLine cmd("~ Fluctus - OpenCL wavefront path tracer ~", ' ', "0.1", false);

        // Leave out version, add help manually
        auto output = cmd.getOutput();
        TCLAP::HelpVisitor v(&cmd, &output);
        TCLAP::SwitchArg help("h", "help", "Displays usage information and exits.", cmd, false, &v);

        TCLAP::ValueArg<int> aWidth("x", "width", "Window width", false, s.getWindowWidth(), "int");
        cmd.add(aWidth);

        TCLAP::ValueArg<int> aHeight("y", "height", "Window height", false, s.getWindowHeight(), "int");
        cmd.add(aHeight);

        TCLAP::ValueArg<int> aSpp("s", "samples", "Samples per pixel to render in batch mode", false, 32, "int");
        cmd.add(aSpp);

        TCLAP::SwitchArg aBatch("b", "batch", "Batch mode", cmd, false);

        TCLAP::UnlabeledMultiArg<std::string> aScenes("Scene", "Scene(s) to render, file selector used if empty", false, "string");
        cmd.add(aScenes);

        // Parse the argv array
        cmd.parse(argc, argv);
        width = aWidth.getValue();
        height = aHeight.getValue();
        spp = aSpp.getValue();
        interactiveMode = !aBatch.getValue();
        scenes = aScenes.getValue();

        if (width < 0)
            throw TCLAP::ArgException("Invalid value", "width");
        if (height < 0)
            throw TCLAP::ArgException("Invalid value", "height");
        if (spp < 0)
            throw TCLAP::ArgException("Invalid value", "samples");
        if (interactiveMode && scenes.size() > 1)
            throw TCLAP::ArgException("Only one scene allowed in interactive mode", "Scene");
    }
    catch (TCLAP::ArgException &e)
    {
        std::cout << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        waitExit();
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
    {
        if (scenes.size() > 0)
            tracer.init(width, height, scenes[0]);
        else
            tracer.init(width, height);
        
        std::cout << "Starting in interactive mode" << std::endl;
        tracer.renderInteractive();
    }
        
    else
    {
        std::cout << "Starting in batch mode" << std::endl;
        for (std::string &scene : scenes)
        {
            tracer.init(width, height, scene);
            tracer.renderSingle(spp);
        }

        if (scenes.size() == 0)
        {
            tracer.init(width, height);
            tracer.renderSingle(spp);
        }
    }
        

    glfwTerminate();

    return 0;
}

