#pragma once

#include "clcontext.hpp"
#include <cmath>
#include "geom.h"
#include "window.hpp"
#include "math/float3.hpp"

using FireRays::float4;

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

    bool paramsUpdatePending = true; // force one param update
};
