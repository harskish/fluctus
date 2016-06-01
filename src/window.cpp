#include "window.hpp"

Window::~Window() {
    if(gl_PBO) glDeleteTextures(1, &gl_PBO);
}

void keyPressCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void errorCallback(int error, const char *desc)
{
    std::cerr << desc << " (error " << error << ")" << std::endl;
}

void windowCloseCallback(GLFWwindow *window)
{
    // Can be delayed by setting value to false temporarily
    // glfwSetWindowShouldClose(window, GL_FALSE);
}

void Window::init() {
    window = glfwCreateWindow(width, height, "HOLDTHEDOOR!", NULL, NULL); // monitor, share
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyPressCallback);
    glfwSetErrorCallback(errorCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);

    createPBO();
}

void Window::repaint() {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    std::cout << "Dimensons from framebuffer: " << w << "x" << h << std::endl;

    /*glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw a single quad
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, gl_texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(0, 1); glVertex2f(0, h);
    glTexCoord2f(1, 1); glVertex2f(w, h);
    glTexCoord2f(1, 0); glVertex2f(w, 0);
    glEnd();*/

    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    /*
     * Set up an ortho-projection such that the bottom/left corner
     * of the image plane is 0,0.
     */
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glDrawPixels(w, h, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// TODO: use FBO/RenderBuffer instead?
void Window::createPBO() {
    if (gl_PBO) {
        //clReleaseMemObject(clPBO); // RELEASE CL_OBJ ALSO!
        std::cout << "Removing old gl_PBO" << std::endl;
        glDeleteBuffers(1, &gl_PBO);
    }

    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    
    // STREAM_DRAW because of frequent updates
    std::cout << "Allocating GL-PBO with " << width * height * sizeof(GLfloat) * 4 << " bytes" << std::endl;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    std::cout << "Created GL-PBO at " << gl_PBO << std::endl;
}




