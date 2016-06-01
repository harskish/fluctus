#pragma once

#include <GLFW/glfw3.h>
#include <iostream>

class Window
{
public:
    Window(int w, int h) : width(w), height(h) { init(); }
    ~Window();

    void repaint();

    bool available()
    { 
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint getPBO() { return gl_PBO; }

private:
    void init();
    void createPBO();

    int width, height;
    GLFWwindow *window;
    GLuint gl_PBO = 0;
};