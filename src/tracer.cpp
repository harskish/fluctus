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
    cam.right = float4(1.0f, 0.0f, 0.0f, 0.0f);
    cam.up = float4(0.0f, 1.0f, 0.0f, 0.0f);
    cam.dir = float4(0.0f, 0.0f, -1.0f, 0.0f);
    cam.fov = 70.0f;
    params.camera = cam;

    camera_rotation = float2(180.0f, 0.0f);
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
    Camera &cam = params.camera;

    switch(key) {
        case GLFW_KEY_W:
            cam.pos += 0.1f * cam.dir;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_A:
            cam.pos -= 0.1f * cam.right;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_S:
            cam.pos -= 0.1f * cam.dir;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_D:
            cam.pos += 0.1f * cam.right;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_R:
            cam.pos += 0.1f * cam.up;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_F:
            cam.pos -= 0.1f * cam.up;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_UP:
            camera_rotation.y += 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_DOWN:
            camera_rotation.y -= 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_LEFT:
            camera_rotation.x += 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_RIGHT:
            camera_rotation.x -= 5.0f;
            paramsUpdatePending = true;
            break;
    }

    // Update camera and other params
    if(paramsUpdatePending) {
        matrix rot = rotation(float3(1, 0, 0), toRad(camera_rotation.y)) * rotation(float3(0, 1, 0), toRad(camera_rotation.x));

        cam.right = float4(rot.m00, rot.m01, rot.m02, 0.0f); // row 1
        cam.up =    float4(rot.m10, rot.m11, rot.m12, 0.0f); // row 2
        cam.dir =   float4(rot.m20, rot.m21, rot.m22, 0.0f); // row 3

        //std::cout << "Up is: " << rot.m10 << ", " << rot.m11 << ", "  << rot.m12 << std::endl;
        std::cout << "Rotation is: " << camera_rotation.x << ", " << camera_rotation.y << std::endl;
    }
}