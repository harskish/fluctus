#include "window.hpp"

Window::~Window()
{
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Window *instance = reinterpret_cast<Window*>(ptr);
    
    instance->createPBO();
    instance->createCLPBO();
}

void windowCloseCallback(GLFWwindow *window)
{
    // Can be delayed by setting value to false temporarily
    // glfwSetWindowShouldClose(window, GL_FALSE);
}

void Window::init(int width, int height)
{
    window = glfwCreateWindow(width, height, "HOLDTHEDOOR!", NULL, NULL); // monitor, share
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyPressCallback);
    glfwSetErrorCallback(errorCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetWindowUserPointer(window, (void*)this);

    createPBO();
}

void Window::getFBSize(int &w, int &h)
{
    glfwGetFramebufferSize(window, &w, &h);
}

void Window::repaint()
{
    int w, h;
    getFBSize(w, h);

    //std::cout << "Dimensons from framebuffer: " << w << "x" << h << std::endl;

    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Set up an ortho-projection such that the bottom/left
    // corner of the image plane is 0,0
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glDrawPixels(w, h, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);
}

// TODO: use FBO/RenderBuffer instead?
void Window::createPBO()
{
    if (gl_PBO) {
        std::cout << "Removing old gl_PBO" << std::endl;
        glDeleteBuffers(1, &gl_PBO);
    }

    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);

    int width, height;
    getFBSize(width, height);
    
    // STREAM_DRAW because of frequent updates
    std::cout << "Allocating GL-PBO with " << width * height * sizeof(GLfloat) * 4 << " bytes" << std::endl;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    std::cout << "Created GL-PBO at " << gl_PBO << std::endl;
}

// Call in createPBO()?
void Window::createCLPBO()
{
    cl_ctx->createPBO(gl_PBO);
}




