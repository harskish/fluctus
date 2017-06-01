#include "window.hpp"
#include "tracer.hpp"

// For keys that need to be registered only once per press
void keyPressCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_RELEASE)
        return;

    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GL_TRUE);

    // Pass keypress to tracer
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);
    instance->handleKeypress(key);
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

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);

    instance->handleMouseButton(button, action);
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) //static?
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);

    instance->handleCursorPos(xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	void *ptr = glfwGetWindowUserPointer(window);
	Tracer *instance = reinterpret_cast<Tracer*>(ptr);

	instance->handleMouseScroll(yoffset);
}

PTWindow::PTWindow(int width, int height, void *tracer)
{
    window = glfwCreateWindow(width, height, "HOLDTHEDOOR!", NULL, NULL); // monitor, share
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSetErrorCallback(errorCallback);
    glfwSetKeyCallback(window, keyPressCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowUserPointer(window, tracer);

    // For key polling
    glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);

    // ===============================================
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "Error: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
        
    std::cout << "Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    // ===============================================

    createTextures();
}

PTWindow::~PTWindow()
{
    if(gl_textures[0]) glDeleteTextures(2, gl_textures);
}

void PTWindow::requestClose()
{
    std::cout << "Setting window close flag..." << std::endl;
    glfwSetWindowShouldClose(window, GL_TRUE);
}

void PTWindow::getFBSize(unsigned int &w, unsigned int &h)
{
    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    w = (unsigned int) fbw;
    h = (unsigned int) fbh;
}

void PTWindow::repaint(int frontBuffer)
{
    unsigned int w, h;
    getFBSize(w, h);
    
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw a single quad
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, gl_textures[frontBuffer]);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(0, 1); glVertex2f(0, h);
    glTexCoord2f(1, 1); glVertex2f(w, h);
    glTexCoord2f(1, 0); glVertex2f(w, 0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glfwSwapBuffers(window);

    if(show_fps)
        calcFPS(1.0, "HOLDTHEDOOR");
}


void PTWindow::drawTexture(int frontBuffer)
{
	unsigned int w, h;
	getFBSize(w, h);
	glViewport(0, 0, w, h);

	glActiveTexture(GL_TEXTURE0); // make texture unit 0 active
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, gl_textures[frontBuffer]);

	float4 posLo(-1.0f, -1.0f, 0.0f, 1.0f);
	float2 posHi(1.0f, 1.0f);
	float2 texLo(0.0f, 0.0f);
	float2 texHi(1.0f, 1.0f);

	// Vertex attributes
	F32 posAttrib[] =
	{
		posLo.x, posLo.y, posLo.z, posLo.w,
		posHi.x, posLo.y, posLo.z, posLo.w,
		posLo.x, posHi.y, posLo.z, posLo.w,
		posHi.x, posHi.y, posLo.z, posLo.w,
	};

	F32 texAttrib[] =
	{
		texLo.x, texLo.y,
		texHi.x, texLo.y,
		texLo.x, texHi.y,
		texHi.x, texHi.y,
	};

	// Create program.
	static const char* progId = "PTWindow::drawTexture";
	GLProgram* prog = GLProgram::get(progId);
	if (!prog)
	{
		prog = new GLProgram(
			GL_SHADER_SOURCE(
				attribute vec4 posAttrib;
				attribute vec2 texAttrib;
				varying vec2 texVarying;
				void main()
				{
					gl_Position = posAttrib;
					texVarying = texAttrib;
				}
			),
			GL_SHADER_SOURCE(
				uniform sampler2D texSampler;
				varying vec2 texVarying;
				void main()
				{
					gl_FragColor = texture2D(texSampler, texVarying);
				}
			)
		);

		// Update static shader storage
		GLProgram::set(progId, prog);
	}

	// Draw image
	prog->use();
	prog->setUniform(prog->getUniformLoc("texSampler"), 0); // texture unit 0
	prog->setAttrib(prog->getAttribLoc("posAttrib"), 4, GL_FLOAT, 0, posAttrib);
	prog->setAttrib(prog->getAttribLoc("texAttrib"), 2, GL_FLOAT, 0, texAttrib);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	prog->resetAttribs();

	glBindTexture(GL_TEXTURE_2D, 0);
	glfwSwapBuffers(window);

	if (show_fps)
		calcFPS(1.0, "HOLDTHEDOOR");
}

// Create front and back buffers
void PTWindow::createTextures()
{
    if (gl_textures[0]) {
        std::cout << "Removing old textures" << std::endl;
        glDeleteTextures(2, gl_textures);
    }

    unsigned int width, height;
    getFBSize(width, height);

    // Size of texture depends on render resolution scale
    float renderScale = Settings::getInstance().getRenderScale();
    this->textureWidth = static_cast<unsigned int>(width * renderScale);
    this->textureHeight = static_cast<unsigned int>(height * renderScale);
    std::cout << "New texture size: " << this->textureWidth << "x" << this->textureHeight << std::endl;

    glGenTextures(2, gl_textures);
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, gl_textures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->textureWidth, this->textureHeight, 0, GL_RGBA, GL_FLOAT, 0);
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

double PTWindow::calcFPS(double interval, std::string theWindowTitle)
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
        float mSps = fps * this->textureWidth * this->textureHeight / 1e6;
 
        // If the user specified a window title to append the FPS value to...
        if (theWindowTitle != "NONE")
        {
            // Convert the fps value into a string using an output stringstream
            std::ostringstream stream;
            stream.precision(2);
            stream << std::fixed << " | FPS: " << fps << " | Samples/sec: " << mSps << "M";
            std::string fpsString = stream.str();
 
            // Append the FPS and samples/sec to the window title details
            theWindowTitle += fpsString;
 
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

float2 PTWindow::getCursorPos()
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return float2((float)xpos, (float)ypos);
}

bool PTWindow::keyPressed(int key)
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}




