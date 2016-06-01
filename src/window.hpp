#pragma once

#include <GLFW/glfw3.h>
#include <iostream>
#include "CLContext.hpp"

class Window
{
public:
    Window(int w, int h) { init(w, h); }
    ~Window();

    void repaint();

    bool available()
    { 
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    void setCLCtx(CLContext *ctx) { cl_ctx = ctx; }
    void createPBO();
    void createCLPBO();

    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint getPBO() { return gl_PBO; }
    void getFBSize(int &w, int &h);

private:
    void init(int w, int h);

    GLFWwindow *window;
    CLContext *cl_ctx;
    GLuint gl_PBO = 0;
};