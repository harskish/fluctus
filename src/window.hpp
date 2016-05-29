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
    GLuint getTexture() { return gl_texture; }

private:
    void init();
    void createTexture();

    GLFWwindow *window;
    GLuint gl_texture;
    unsigned int width, height;
};