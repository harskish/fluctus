#pragma once

#include <GL/glew.h> // somehow included first of all glew includes?
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include "clcontext.hpp"
#include "math/float2.hpp"

using FireRays::float2;

class PTWindow
{
public:
    PTWindow(int w, int h, void *tracer);
    ~PTWindow();

    void repaint(int frontBuffer);

    bool available()
    {
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    void setShowFPS(bool show) { show_fps = show; }
    void createTextures();
    void requestClose();

    float2 getCursorPos();
    bool keyPressed(int key);

    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint *getTexPtr() { return gl_textures; }
    void getFBSize(unsigned int &w, unsigned int &h);

private:
    double calcFPS(double interval = 1.0, std::string theWindowTitle = "NONE");

    GLFWwindow *window;
    GLuint gl_textures[2] = { 0, 0 };
    unsigned int textureWidth, textureHeight;
    bool show_fps = false;
};