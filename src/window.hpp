#pragma once
#include <GLFW/glfw3.h>
#include <iostream>

class Window
{
public:
    Window(const unsigned int w, const unsigned int h) : width(w), height(h) { init(); }
    ~Window();

    bool available() { return alive; }

private:
    void init();

    GLFWwindow *window;
    bool alive = false;
    unsigned int width, height;
};