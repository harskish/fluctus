#pragma once

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include "CLContext.hpp"

class Window
{
public:
    Window(int w, int h, void *tracer);
    ~Window();

    void repaint();

    bool available()
    {
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    void setShowFPS(bool show) { show_fps = show; }
    void createPBO();

    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint getPBO() { return gl_PBO; }
    void getFBSize(unsigned int &w, unsigned int &h);

private:
    double calcFPS(double interval = 1.0, std::string theWindowTitle = "NONE");

    GLFWwindow *window;
    GLuint gl_PBO = 0;
    bool show_fps = false;
};