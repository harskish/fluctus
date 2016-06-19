#include "window.hpp"
#include "tracer.hpp"

void keyPressCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(action == GLFW_RELEASE)
        return;

    if(key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GL_TRUE);

    // Pass keypress to tracer
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *tracer = reinterpret_cast<Tracer*>(ptr);
    tracer->handleKeypress(key);
}

void errorCallback(int error, const char *desc)
{
    std::cerr << desc << " (error " << error << ")" << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);
    
    instance->resizeBuffers();
}

void windowCloseCallback(GLFWwindow *window)
{
    // Can be delayed by setting value to false temporarily
    // glfwSetWindowShouldClose(window, GL_FALSE);
}

Window::Window(int width, int height, void *tracer)
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
    glfwSetWindowUserPointer(window, tracer); // for callbacks

    createPBO();
}

Window::~Window()
{
    if(gl_PBO) glDeleteTextures(1, &gl_PBO);
}

void Window::getFBSize(unsigned int &w, unsigned int &h)
{
    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    w = (unsigned int) fbw;
    h = (unsigned int) fbh;
}

void Window::repaint()
{
    unsigned int w, h;
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

    if(show_fps)
        calcFPS(1.0, "HOLDTHEDOOR");
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

    unsigned int width, height;
    getFBSize(width, height);

    std::cout << "New size: " << width << "x" << height << std::endl;
    
    // STREAM_DRAW because of frequent updates
    std::cout << "Allocating GL-PBO with " << width * height * sizeof(GLfloat) * 4 << " bytes" << std::endl;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    std::cout << "Created GL-PBO at " << gl_PBO << std::endl;
}

double Window::calcFPS(double interval, std::string theWindowTitle)
{
    // Static values, only initialised once
    static double tLast      = glfwGetTime();
    static int    frameCount = 0;
    static double fps        = 0.0;
 
    // Current time in seconds since the program started
    double tNow = glfwGetTime();
 
    // Sanity check
    interval = std::max(0.1, std::min(interval, 10.0));
 
    // Time to show FPS?
    if ((tNow - tLast) > interval)
    {
        fps = (double)frameCount / (tNow - tLast);
 
        // If the user specified a window title to append the FPS value to...
        if (theWindowTitle != "NONE")
        {
            // Convert the fps value into a string using an output stringstream
            std::ostringstream stream;
            stream.precision(3);
            stream << std::fixed << fps;
            std::string fpsString = stream.str();            
 
            // Append the FPS value to the window title details
            theWindowTitle += " | FPS: " + fpsString;
 
            // Convert the new window title to a c_str and set it
            const char* pszConstString = theWindowTitle.c_str();
            glfwSetWindowTitle(window, pszConstString);
        }
        else
        {
            std::cout << "FPS: " << fps << std::endl;
        }
 
        // Reset counter and time
        frameCount = 0;
        tLast = glfwGetTime();
    }
    else
    {
        frameCount++;
    }
 
    return fps;
}




