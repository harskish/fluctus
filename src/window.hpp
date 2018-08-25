#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>
#include <iostream>
#include <string>
#include "GLProgram.hpp"
#include "math/float2.hpp"

using FireRays::float2;

class ProgressView;
class CLContext;
class PTWindow
{
public:
    PTWindow(int w, int h, void *tracer);
    ~PTWindow();

    enum RenderMethod
    {
        WAVEFRONT,
        MEGAKERNEL
    };
    
    void draw();
    void setRenderMethod(RenderMethod method) { renderMethod = method; };
    void setFrontBuffer(int fb) { frontBuffer = fb; }

    bool available()
    {
        return !glfwWindowShouldClose(window);
    }

    void setShowFPS(bool show) { show_fps = show; }
    void createTextures();
	void createPBO();
    void requestClose();
    void setupGUI();

    void showError(const std::string &msg);
    void showMessage(const std::string &primary, const std::string &secondary = "");
    void hideMessage();

    float2 getCursorPos();
    bool keyPressed(int key);

    void setCLContextPtr(CLContext *ptr) { clctx = ptr; }
    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint *getTexPtr() { return gl_textures; }
	GLuint getPBO() { return gl_PBO; }
    void getFBSize(unsigned int &w, unsigned int &h); // unscaled FB
    nanogui::Screen *getScreen() { return screen; }
    ProgressView *getProgressView() { return progress; }

    // Internal rendering resolution
    unsigned int getTexWidth() { return textureWidth; }
    unsigned int getTexHeight() { return textureHeight; }

private:
    void drawPixelBuffer();
    void drawTexture();
    double calcFPS(double interval = 1.0, std::string theWindowTitle = "");

    CLContext *clctx; // for showing stats
    GLFWwindow *window;
    nanogui::Screen *screen;
    ProgressView *progress = nullptr;

    GLuint gl_textures[2] = { 0, 0 };
	GLuint gl_PBO = 0;
	GLuint gl_PBO_texture;
    unsigned int textureWidth = 0, textureHeight = 0;
    bool show_fps = false;
    RenderMethod renderMethod = WAVEFRONT;
    int frontBuffer = 0;
};