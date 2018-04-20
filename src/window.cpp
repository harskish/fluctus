#include "window.hpp"
#include "tracer.hpp"
#include "progressview.hpp"

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
    instance->handleKeypress(key, scancode, action, mods);
}

void errorCallback(int error, const char *desc)
{
    std::cerr << desc << " (error " << error << ")" << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);
    
    instance->resizeBuffers(width, height);
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

    instance->handleMouseButton(button, action, mods);
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) //static?
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);

    instance->handleCursorPos(xpos, ypos);
}

void drop_callback(GLFWwindow *window, int count, const char **filenames)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);

    instance->handleFileDrop(count, filenames);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	void *ptr = glfwGetWindowUserPointer(window);
	Tracer *instance = reinterpret_cast<Tracer*>(ptr);

	instance->handleMouseScroll(yoffset);
}

void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    void *ptr = glfwGetWindowUserPointer(window);
    Tracer *instance = reinterpret_cast<Tracer*>(ptr);

    instance->handleChar(codepoint);
}

PTWindow::PTWindow(int width, int height, void *tracer)
{
    // Modern OpenGL (3.3), core context, no backwards compatibility
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window = glfwCreateWindow(width, height, "Fluctus", NULL, NULL); // monitor, share
    if (!window) {
        glfwTerminate();
        std::cout << "Could not create GLFW window" << std::endl;
        waitExit();
    }

    glfwMakeContextCurrent(window);
    glfwSetErrorCallback(errorCallback);
    glfwSetKeyCallback(window, keyPressCallback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetCharCallback(window, char_callback);
    glfwSetDropCallback(window, drop_callback);
    glfwSetWindowUserPointer(window, tracer);

    // For key polling
    glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    printf("<OpenGL> Version: %s, GLSL: %s\n", glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

    createTextures();
	createPBO();
}

void PTWindow::setupGUI()
{
    // Save current size
    int w, h;
    glfwGetWindowSize(window, &w, &h);

    // Create a nanogui screen and pass the glfw pointer to initialize
    screen = new nanogui::Screen();
    screen->initialize(window, true);

    // Revert size changes done by NanoGUI
    glfwSetWindowSize(window, w, h);
    glfwPollEvents(); // makes window appear on MacOS

    // Setup progress view
    progress = new ProgressView(screen);
    progress->setRenderFunc([&]() { draw(); });
}

void PTWindow::showError(const std::string & msg)
{
    progress->showError(msg);
}

void PTWindow::showMessage(const std::string & primary, const std::string & secondary /* = "" */)
{
    progress->showMessage(primary, secondary);
}

void PTWindow::hideMessage()
{
    progress->hide();
}


PTWindow::~PTWindow()
{
    if (gl_textures[0]) glDeleteTextures(2, gl_textures);
	if (gl_PBO) glDeleteBuffers(1, &gl_PBO);
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

void PTWindow::draw()
{
    switch (renderMethod)
    {
    case WAVEFRONT:
        drawPixelBuffer();
        break;
    case MICROKERNEL:
        drawPixelBuffer();
        break;
    default:
        std::cout << "Invalid render method!" << std::endl;
        break;
    }
}

void PTWindow::setSize(int w, int h)
{
    screen->setSize({ w, h });
    glfwSetWindowSize(window, w, h);
    if (progress) progress->center();
}

// https://devtalk.nvidia.com/default/topic/541646/opengl/draw-pbo-into-the-screen-performance/
void PTWindow::drawPixelBuffer()
{
	unsigned int w, h;
	getFBSize(w, h);
	glViewport(0, 0, w, h);

	glActiveTexture(GL_TEXTURE0); // make texture unit 0 active
	glBindTexture(GL_TEXTURE_2D, gl_PBO_texture);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)this->textureWidth, (GLsizei)this->textureHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// Vertex attributes
	const GLfloat posAttrib[] =
	{
        -1.0f, -1.0f, 0.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
		 1.0f,  1.0f, 0.0f, 1.0f,
	};

    const GLfloat texAttrib[] =
	{
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
	};

	// Create program.
	static const char* progId = "PTWindow::drawPBO";
	GLProgram* prog = GLProgram::get(progId);
	if (!prog)
	{
		prog = new GLProgram(
            "#version 330\n"
			GL_SHADER_SOURCE(
                layout (location = 0) in vec4 posAttrib;
                layout (location = 1) in vec2 texAttrib;
				out vec2 texVarying;
				void main()
				{
					gl_Position = posAttrib;
					texVarying = texAttrib;
				}
			),
            "#version 330\n"
			GL_SHADER_SOURCE(
				uniform sampler2D texSampler;
				in vec2 texVarying;
                out vec4 fragColor;

				bool isnan4( vec4 val )
				{
					for (int i = 0; i < 4; i++)
						if ( !(val[i] < 0.0 || 0.0 < val[i] || val[i] == 0.0 ) ) return true;

					return false;
				}

                bool isinf4( vec4 val )
                {
                    for (int i = 0; i < 4; i++)
                        if ( val[i] != val[i] ) return true;

                    return false;
                }

				void main()
				{
                    // Texture contains tonemapped and gamma-corrected data
					vec4 color = texture(texSampler, texVarying);
                    fragColor = color;
				}
			)
		);

		// Update static shader storage
		GLProgram::set(progId, prog);

        // Setup VAO
        GLuint vbo[2], vao[1];
        glGenBuffers(2, vbo); // VBO: stores single attribute (pos/color/normal etc.)
        glGenVertexArrays(1, vao); // VAO: bundles VBO:s together
        glBindVertexArray(vao[0]);
        GLcheckErrors();

        // Positions
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), posAttrib, GL_STATIC_DRAW); // STREAM: modified once, used many times
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0); // coordinate data in attribute index 0, four floats per vertex
        glEnableVertexAttribArray(0); // enable index 0 within VAO
        GLcheckErrors();

        // Texture coordinates
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texAttrib, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0); // TODO: normalized = false...?
        glEnableVertexAttribArray(1); // enable index 1 within VAO
        GLcheckErrors();

        prog->addVAOs(vao, 1);
	}

	// Draw image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	prog->use();
	prog->setUniform(prog->getUniformLoc("texSampler"), 0); // texture unit 0
    prog->bindVAO(0);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    GLcheckErrors();

    // Draw nanogui
    screen->drawContents();
    screen->drawWidgets();
    
	glfwSwapBuffers(window);

	if (show_fps)
		calcFPS(1.0, "Fluctus");
}


void PTWindow::drawTexture()
{
	unsigned int w, h;
	getFBSize(w, h);
	glViewport(0, 0, w, h);

	glActiveTexture(GL_TEXTURE0); // make texture unit 0 active
	glBindTexture(GL_TEXTURE_2D, gl_textures[frontBuffer]);

    const GLfloat posAttrib[] =
    {
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,
    };

    const GLfloat texAttrib[] =
    {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

	// Create program.
	static const char* progId = "PTWindow::drawTexture";
	GLProgram* prog = GLProgram::get(progId);
	if (!prog)
	{
		prog = new GLProgram(
            "#version 330\n"
            GL_SHADER_SOURCE(
                layout (location = 0) in vec4 posAttrib;
                layout (location = 1) in vec2 texAttrib;
                out vec2 texVarying;
                void main()
                {
                    gl_Position = posAttrib;
                    texVarying = texAttrib;
                }
			),
            "#version 330\n"
			GL_SHADER_SOURCE(
				uniform sampler2D texSampler;
				in vec2 texVarying;
                out vec4 fragColor;
                
                bool isnan4(vec4 val)
                {
                    for (int i = 0; i < 4; i++)
                        if (!(val[i] < 0.0 || 0.0 < val[i] || val[i] == 0.0)) return true;

                    return false;
                }

                bool isinf4(vec4 val)
                {
                    for (int i = 0; i < 4; i++)
                        if (val[i] != val[i]) return true;

                    return false;
                }

                void main()
                {
                    vec4 color = texture(texSampler, texVarying);
                    if (color.a > 0.0)
                        color = color / color.a;

                    if (isnan4(color))
                        color = vec4(1.0, 0.0, 1.0, 1.0);
                    if (isinf4(color))
                        color = vec4(0.0, 1.0, 1.0, 1.0);

                    // Gamma correction
                    color.xyz = pow(color.xyz, vec3(1.0 / 2.2));

                    fragColor = color;
                }
			)
		);

		// Update static shader storage
		GLProgram::set(progId, prog);

        // Setup VAO
        GLuint vbo[2], vao[1];
        glGenBuffers(2, vbo); // VBO: stores single attribute (pos/color/normal etc.)
        glGenVertexArrays(1, vao); // VAO: bundles VBO:s together
        glBindVertexArray(vao[0]);
        GLcheckErrors();

        // Positions
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), posAttrib, GL_STATIC_DRAW); // STREAM: modified once, used many times
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0); // coordinate data in attribute index 0, four floats per vertex
        glEnableVertexAttribArray(0); // enable index 0 within VAO
        GLcheckErrors();

        // Texture coordinates
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texAttrib, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0); // TODO: normalized = false...?
        glEnableVertexAttribArray(1); // enable index 1 within VAO
        GLcheckErrors();

        prog->addVAOs(vao, 1);
	}

    // Draw image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    prog->use();
    prog->setUniform(prog->getUniformLoc("texSampler"), 0); // texture unit 0
    prog->bindVAO(0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    GLcheckErrors();

    // Draw nanogui
    screen->drawContents();
    screen->drawWidgets();

	glfwSwapBuffers(window);

	if (show_fps)
		calcFPS(1.0, "Fluctus");
}

void PTWindow::setShowFPS(bool show)
{
    show_fps = show;
    if (!show) glfwSetWindowTitle(window, "Fluctus");
}

// Create front and back buffers
void PTWindow::createTextures()
{
    unsigned int width, height;
    getFBSize(width, height);

    // Size of texture depends on render resolution scale
    float renderScale = Settings::getInstance().getRenderScale();
    unsigned int texWidth = static_cast<unsigned int>(width * renderScale);
    unsigned int texHeight = static_cast<unsigned int>(height * renderScale);
    
    if (texWidth == this->textureWidth && texHeight == this->textureHeight)
        return;

    this->textureWidth = texWidth;
    this->textureHeight = texHeight;
    std::cout << "New texture size: " << this->textureWidth << "x" << this->textureHeight << std::endl;

    if (gl_textures[0])
        glDeleteTextures(2, gl_textures);

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

// Create pixel buffer object used by microkernels
void PTWindow::createPBO()
{
	if (gl_PBO) {
		glDeleteBuffers(1, &gl_PBO);
		glDeleteTextures(1, &gl_PBO_texture);
	}

	// Size of texture depends on render resolution scale
	unsigned int width, height;
	getFBSize(width, height);
	float renderScale = Settings::getInstance().getRenderScale();
	this->textureWidth = static_cast<unsigned int>(width * renderScale);
	this->textureHeight = static_cast<unsigned int>(height * renderScale);

	// STREAM_DRAW because of frequent updates
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, this->textureWidth * this->textureHeight * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// Create GL-only texture for PBO displaying
	glGenTextures(1, &gl_PBO_texture);
	glBindTexture(GL_TEXTURE_2D, gl_PBO_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
        
        const PerfNumbers perf = clctx->getRenderPerf();
        float MRps = perf.total;
 
        // If the user specified a window title to append the FPS value to...
        if (theWindowTitle.length() > 0)
        {
            // Convert the fps value into a string using an output stringstream
            std::ostringstream stream;
            stream.precision(2);
            stream << std::fixed << " | FPS: " << fps << " | Rays/s: " << MRps << "M";
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




