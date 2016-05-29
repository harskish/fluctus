#pragma once
#include <GLFW/glfw3.h>
#include <iostream>

class Window
{
public:
    Window(const unsigned int w, const unsigned int h) : width(w), height(h) { init(); }
    ~Window();

    bool available()
    { 
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    GLFWwindow *glfwWindowPtr() { return window; }

private:
    void init();

    GLFWwindow *window;
    //bool alive = false;
    unsigned int width, height;
};