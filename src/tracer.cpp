#include "tracer.hpp"
#include "geom.h"

Tracer::Tracer(int width, int height)
{
    scene = new Scene("assets/torus.obj");
    this->constructHierarchy(scene->getTriangles(), SplitMode_Sah);

    window = new PTWindow(width, height, this);
    window->setShowFPS(true);
    clctx = new CLContext(window->getPBO());

    params.width = (unsigned int)width;
    params.height = (unsigned int)height;
    params.n_lights = sizeof(test_lights) / sizeof(Light);
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);

    Camera cam;
    cam.pos = float4(0.0f, 1.0f, 3.5f, 0.0f);
    cam.right = float4(1.0f, 0.0f, 0.0f, 0.0f);
    cam.up = float4(0.0f, 1.0f, 0.0f, 0.0f);
    cam.dir = float4(0.0f, 0.0f, -1.0f, 0.0f);
    cam.fov = 60.0f;
    params.camera = cam;
    cameraRotation = float2(-180.0f, 0.0f);
}

Tracer::~Tracer()
{
    delete window;
    delete clctx;
    delete scene;
    delete bvh;
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
    if(paramsUpdatePending)
    {
        clctx->updateParams(params);
        paramsUpdatePending = false;
    }

    // Advance render state
    clctx->executeKernel(params);

    // Draw progress to screen
    window->repaint();
}

void Tracer::loadHierarchy(const char* filename, std::vector<RTTriangle>& triangles)
{
    m_triangles = &triangles;
    bvh = new BVH(m_triangles, filename);
}

void Tracer::saveHierarchy(const char* filename)
{
    bvh->exportTo(filename);
}

void Tracer::constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode)
{
    m_triangles = &triangles;
    bvh = new BVH(m_triangles, splitMode);
}

void Tracer::updateCamera()
{
    if(cameraRotation.x < 0) cameraRotation.x += 360.0f;
    if(cameraRotation.y < 0) cameraRotation.y += 360.0f;
    if(cameraRotation.x > 360.0f) cameraRotation.x -= 360.0f;
    if(cameraRotation.y > 360.0f) cameraRotation.y -= 360.0f;

    matrix rot = rotation(float3(1, 0, 0), toRad(cameraRotation.y)) * rotation(float3(0, 1, 0), toRad(cameraRotation.x));

    params.camera.right = float4(rot.m00, rot.m01, rot.m02, 0.0f); // row 1
    params.camera.up =    float4(rot.m10, rot.m11, rot.m12, 0.0f); // row 2
    params.camera.dir =   float4(rot.m20, rot.m21, rot.m22, 0.0f); // row 3
}

void Tracer::handleKeypress(int key)
{
    Camera &cam = params.camera;

    switch(key)
    {
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
            cameraRotation.y += 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_DOWN:
            cameraRotation.y -= 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_LEFT:
            cameraRotation.x += 5.0f;
            paramsUpdatePending = true;
            break;
        case GLFW_KEY_RIGHT:
            cameraRotation.x -= 5.0f;
            paramsUpdatePending = true;
            break;
    }

    // Update camera and other params
    if(paramsUpdatePending)
    {
        updateCamera();
    }
}

void Tracer::handleMouseButton(int key, int action)
{
    switch(key)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            if(action == GLFW_PRESS)
            {
                lastCursorPos = window->getCursorPos();
                mouseButtonState[0] = true;
                //std::cout << "Left mouse button pressed" << std::endl;
            }
            if(action == GLFW_RELEASE)
            {
                mouseButtonState[0] = false;
                //std::cout << "Left mouse button released" << std::endl;
            }
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            if(action == GLFW_PRESS) mouseButtonState[1] = true;
            if(action == GLFW_RELEASE) mouseButtonState[2] = false;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            if(action == GLFW_PRESS) mouseButtonState[2] = true;
            if(action == GLFW_RELEASE) mouseButtonState[2] = false;
            break;
    }
}

void Tracer::handleCursorPos(double x, double y)
{
    if(mouseButtonState[0])
    {
        float2 newPos = float2((float)x, (float)y);
        float2 delta = lastCursorPos - newPos;

        // std::cout << "Mouse delta: " << delta.x <<  ", " << delta.y << std::endl;

        cameraRotation += delta;
        lastCursorPos = newPos;

        updateCamera();
        paramsUpdatePending = true;
    }
}