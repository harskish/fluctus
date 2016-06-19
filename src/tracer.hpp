#pragma once

#include "clcontext.hpp"
#include <cmath>
#include "geom.h"
#include "window.hpp"

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
};
