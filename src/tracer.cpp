#include "tracer.hpp"
#include "geom.h"

Tracer::Tracer(int width, int height)
{
    window = new Window(width, height, this);
    window->setShowFPS(true);
    clctx = new CLContext(window->getPBO());

    params.width = (unsigned int)width;
    params.height = (unsigned int)height;
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);

    Camera cam;
    cam.pos = float4(0.0f, 0.1f, 2.5f, 0.0f);
    cam.dir = float4(0.0f, -0.2f, -1.0f, 0.0f);
    cam.fov = 70.0f;
    params.camera = cam;
}

Tracer::~Tracer()
{
    delete window;
    delete clctx;
}

bool Tracer::running()
{
    return window->available();
}

// Callback for when the window size changes
void Tracer::resizeBuffers()
{
    window->createPBO();
    clctx->createPBO(window->getPBO());
    paramsUpdatePending = true;
    std::cout << std::endl;
}

void Tracer::update()
{
    // React to key presses
    glfwPollEvents();

    // Update RenderParams in GPU memory if needed
    window->getFBSize(params.width, params.height);
    if(paramsUpdatePending) {
        clctx->updateParams(params);
        paramsUpdatePending = false;
    }

    // Advance render state
    clctx->executeKernel(params);

    // Draw progress to screen
    window->repaint();
}

void Tracer::handleKeypress(int key)
{
    switch(key) {
        case GLFW_KEY_W:
            params.camera.pos.z -= 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_A:
            params.camera.pos.x -= 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_S:
            params.camera.pos.z += 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_D:
            params.camera.pos.x += 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_R:
            params.camera.pos.y += 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_F:
            params.camera.pos.y -= 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_UP:
            params.camera.dir.y += 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_DOWN:
            params.camera.dir.y -= 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_LEFT:
            params.camera.dir.x -= 0.1f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_RIGHT:
            params.camera.dir.x += 0.1f;
            paramsUpdatePending = true;
            break;
    }
}