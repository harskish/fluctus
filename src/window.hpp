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
    GLuint getTexture() { return gl_texture; }

private:
    void init();
    void createTexture();

    int width, height;
    GLFWwindow *window;
    GLuint gl_texture = 0;
};