#include "window.hpp"

Window::~Window() {
    if(gl_texture) glDeleteTextures(1, &gl_texture);
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

    createTexture();
}

/* https://software.intel.com/sites/default/files/managed/da/ab/OpenGLInterop.pdf */
void Window::createTexture() {
    // Remove old texture, useful e.g. when resizing window
    if(gl_texture) {
        std::cout << "Removing old GL-texture" << std::endl;
        glDeleteTextures(1, &gl_texture);
        gl_texture = 0;
    }

    // Generate new texture
    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, 0);
    std::cout << "Created GL-texture at " << gl_texture << std::endl;
}




