#pragma once

#include <GL/glew.h>
#include "clcontext.hpp"
#include <string>
#include <cmath>
#include "geom.h"
#include "window.hpp"
#include "math/float2.hpp"
#include "math/float3.hpp"
#include "math/matrix.hpp"
#include "bvh.hpp"
#include "scene.hpp"

using namespace FireRays;

class Tracer
{
public:
    Tracer(int width, int height);
    ~Tracer();

    bool running();
    void update();
    void resizeBuffers();
    void handleKeypress(int key);
    void handleMouseButton(int key, int action);
    void handleCursorPos(double x, double y);

    void updateCamera();

    // Create/load/export BVH
    void loadHierarchy(const char* filename, std::vector<RTTriangle> &triangles);
    void saveHierarchy(const char* filename);
    void constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode);

private:
    PTWindow *window;
    CLContext *clctx;
    RenderParams params;    // copied into GPU memory
    float2 cameraRotation;  // not passed to GPU but needed for camera basis vectors
    float2 lastCursorPos;
    bool mouseButtonState[3] = { false, false, false };
    bool paramsUpdatePending = true; // force initial param update

    Scene *scene;
    BVH *bvh;
    std::vector<RTTriangle>* m_triangles;
};
