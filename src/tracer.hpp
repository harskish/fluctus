#pragma once

#include "clcontext.hpp"
#include <cmath>
#include "geom.h"
#include "window.hpp"
#include "math/float2.hpp"
#include "math/float3.hpp"
#include "math/matrix.hpp"

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

private:
    Window *window;
    CLContext *clctx;
    RenderParams params;
    float2 camera_rotation; // not passed to GPU but needed for camera basis vectors

    bool paramsUpdatePending = true; // force one param update
};
