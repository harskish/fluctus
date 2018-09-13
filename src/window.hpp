#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>
#include <iostream>
#include <string>
#include "GLProgram.hpp"
#include "math/float2.hpp"

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
        MICROKERNEL
    };
    
    void draw();
    void drawDenoised(); // draw denoised result, cache result
    void displayDenoised(); // display cached result
    void setRenderMethod(RenderMethod method) { renderMethod = method; };
    void setFrontBuffer(int fb) { frontBuffer = fb; }
    void setSize(int w, int h);

    bool available()
    {
        return !glfwWindowShouldClose(window);
    }

    void setShowFPS(bool show);
    void createTextures();
	void createPBOs();
    void requestClose();
    void setupGUI();

    void showError(const std::string &msg);
    void showMessage(const std::string &primary, const std::string &secondary = "");
    void hideMessage();

    FireRays::float2 getCursorPos();
    bool keyPressed(int key);

    void setCLContextPtr(CLContext *ptr) { clctx = ptr; }
    GLFWwindow *glfwWindowPtr() { return window; }
    GLuint *getTexPtr() { return gl_textures; }
	GLuint getPBO() { return gl_PBO; }
    GLuint getAlbedoPBO() { return gl_albedoPBO; }
    GLuint getNormalPBO() { return gl_normalPBO; }
    void getFBSize(unsigned int &w, unsigned int &h); // unscaled FB
    nanogui::Screen *getScreen() { return screen; }
    ProgressView *getProgressView() { return progress; }

    // Internal rendering resolution
    unsigned int getTexWidth() { return textureWidth; }
    unsigned int getTexHeight() { return textureHeight; }

private:
    void drawPixelBuffer(GLuint sourcePBO, GLuint targetTex);
    void drawTexture(GLuint texId);
    double calcFPS(double interval = 1.0, std::string theWindowTitle = "");

    CLContext *clctx; // for showing stats
    GLFWwindow *window;
    nanogui::Screen *screen;
    ProgressView *progress = nullptr;

    GLuint gl_textures[2] = { 0, 0 };
	GLuint gl_PBO = 0;
	GLuint gl_albedoPBO = 0;
	GLuint gl_normalPBO = 0;
	GLuint gl_PBO_texture = 0;
    GLuint gl_denoised_texture = 0;
    unsigned int textureWidth = 0, textureHeight = 0;
    bool show_fps = false;
    RenderMethod renderMethod = WAVEFRONT;
    int frontBuffer = 0;
};