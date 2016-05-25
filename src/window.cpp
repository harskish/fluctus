#include "window.hpp"

Window::~Window() {
    // window destroyed in closeCallback
}

void errorCallback(int error, const char *desc) {
    std::cerr << desc << " (error " << error << ")" << std::endl;
}

void windowCloseCallback(GLFWwindow *win) {
    std::cout << "Terminating..." << std::endl;
    
    // Set close flag to false if further processing needs to be done...

    bool *alive = reinterpret_cast<bool*>(glfwGetWindowUserPointer(win));
    *alive = false;

    glfwDestroyWindow(win);
}

void Window::init() {
    window = glfwCreateWindow(width, height, "HOLDTHEDOOR!", NULL, NULL); // monitor, share
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    alive = true;
    glfwSetWindowUserPointer(window, &alive); // for glfw-callbacks
    glfwMakeContextCurrent(window);
    glfwSetErrorCallback(errorCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
}