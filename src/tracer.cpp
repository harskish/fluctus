#include "tracer.hpp"
#include "geom.h"

inline void calcRps(const unsigned int numRays, double launchTime)
{
    static double lastPrinted = 0;
    double now = glfwGetTime();
    if (now - lastPrinted > 1.0)
    {
        lastPrinted = now;
        double mRps = numRays * 1e-6 / launchTime;
        printf("\rKernel rays/s: %.0fM", mRps);
    }
}

void Tracer::update()
{
    // React to key presses
    glfwPollEvents();
    pollKeys();

    // Update RenderParams in GPU memory if needed
    if(paramsUpdatePending)
    {
        // Update render dimensions
        const float renderScale = Settings::getInstance().getRenderScale();
        window->getFBSize(params.width, params.height);
        params.width = static_cast<unsigned int>(params.width * renderScale);
        params.height = static_cast<unsigned int>(params.height * renderScale);

        clctx->updateParams(params);
        paramsUpdatePending = false;
        iteration = 0; // accumulation reset
    }

    if (useMK)
    {
        glFinish(); // locks execution to refresh rate of display (GL)

        double kStart = glfwGetTime();

        // Generate new camera rays
        clctx->executeRayGenKernel(params);

        // Trace rays
        clctx->executeNextVertexKernel(params);

        // Splat results
        clctx->executeSplatKernel(params, frontBuffer, iteration);

        // TODO: add statistic tracking from kernels
        calcRps(params.width * params.height * 1, glfwGetTime() - kStart);
    }
    else
    {
        glFinish();

        double kStart = glfwGetTime();

        // Advance render state
        clctx->executeMegaKernel(params, frontBuffer, iteration);

        // Primary and extension rays (no shadow rays). Not same as Samples/sec
        calcRps(params.width * params.height * (1 + params.maxBounces), glfwGetTime() - kStart);
    }

    // Draw progress to screen
    window->repaint(frontBuffer);
    frontBuffer = 1 - frontBuffer;

    // Update iteration counter
    iteration++;
}

Tracer::Tracer(int width, int height) : useMK(true)
{
    // done only once (VS debugging stops working if context is recreated)
    window = new PTWindow(width, height, this); // this = glfw user pointer
    window->setShowFPS(true);
    clctx = new CLContext(window->getTexPtr());
    initCamera();
    initAreaLight();
    loadState(); // useful when debugging

    // done whenever a new scene is selected
    init(width, height);
}

// Run whenever a scene is laoded
void Tracer::init(int width, int height)
{
    float renderScale = Settings::getInstance().getRenderScale();

    params.width = static_cast<unsigned int>(width * renderScale);
    params.height = static_cast<unsigned int>(height * renderScale);
    params.n_lights = sizeof(test_lights) / sizeof(PointLight);
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);
    params.useEnvMap = 0;
    params.flashlight = 0;
    params.maxBounces = 2;

    selectScene();
    initEnvMap();
    initHierarchy();

    clctx->uploadSceneData(bvh, scene);

    // Data uploaded to GPU => no longer needed
    delete scene;
    delete bvh;
}

void Tracer::selectScene()
{
    char const * pattern[] = { "*.obj", "*.ply" };
    char const *files = tinyfd_openFileDialog("Select a scene file", "assets/", 2, pattern, NULL, 0); // allow only single selection

    std::string selected = (files) ? std::string(files) : "assets/teapot.ply";
    scene = new Scene(selected);
}

void Tracer::initEnvMap()
{
    EnvironmentMap *envMap = scene->getEnvMap();
    if (envMap)
    {
        params.useEnvMap = 1;
        clctx->createEnvMap(envMap->getData(), envMap->getWidth(), envMap->getHeight());
    }
}

// Check if old hierarchy can be reused
void Tracer::initHierarchy()
{
    std::string hashFile = "hierarchies/hierarchy-" + scene->hashString() + ".bin" ;
    std::ifstream input(hashFile, std::ios::in);

    if (input.good())
    {
        std::cout << "Reusing BVH..." << std::endl;
        loadHierarchy(hashFile, scene->getTriangles());
    }
    else
    {
        std::cout << "Building BVH..." << std::endl;
        constructHierarchy(scene->getTriangles(), SplitMode_Sah);
        saveHierarchy(hashFile);
    }
}

Tracer::~Tracer()
{
    delete window;
    delete clctx;
}

bool Tracer::running()
{
    return window->available();
}

// Callback for when the window size changes
void Tracer::resizeBuffers()
{
    window->createTextures();
    clctx->createTextures(window->getTexPtr());
    paramsUpdatePending = true;
    std::cout << std::endl;
}

inline void writeVec(std::ofstream &out, FireRays::float3 &vec)
{
    write(out, vec.x);
    write(out, vec.y);
    write(out, vec.z);
}

void Tracer::saveState()
{
    std::ofstream out("state.dat", std::ios::binary);

    // Write camera state to file
    if (out.good())
    {
        write(out, cameraRotation.x);
        write(out, cameraRotation.y);
        write(out, params.camera.fov);
        writeVec(out, params.camera.dir);
        writeVec(out, params.camera.pos);
        writeVec(out, params.camera.right);
        writeVec(out, params.camera.up);
        std::cout << "Camera state exported" << std::endl;

        writeVec(out, params.areaLight.N);
        writeVec(out, params.areaLight.pos);
        writeVec(out, params.areaLight.right);
        writeVec(out, params.areaLight.up);
        write(out, params.areaLight.size.x);
        write(out, params.areaLight.size.y);
        std::cout << "AreaLight state exported" << std::endl;
    }
    else
    {
        std::cout << "Could not create state file" << std::endl;
    }
}

inline void readVec(std::ifstream &in, FireRays::float3 &vec)
{
    read(in, vec.x);
    read(in, vec.y);
    read(in, vec.z);
}

void Tracer::loadState()
{
    std::ifstream in("state.dat");
    if (in.good())
    {
        read(in, cameraRotation.x);
        read(in, cameraRotation.y);
        read(in, params.camera.fov);
        readVec(in, params.camera.dir);
        readVec(in, params.camera.pos);
        readVec(in, params.camera.right);
        readVec(in, params.camera.up);
        std::cout << "Camera state imported" << std::endl;

        readVec(in, params.areaLight.N);
        readVec(in, params.areaLight.pos);
        readVec(in, params.areaLight.right);
        readVec(in, params.areaLight.up);
        read(in, params.areaLight.size.x);
        read(in, params.areaLight.size.y);
        std::cout << "AreaLight state imported" << std::endl;
    }
    else
    {
        std::cout << "State file not found" << std::endl;
    }
}

void Tracer::loadHierarchy(const std::string filename, std::vector<RTTriangle>& triangles)
{
    m_triangles = &triangles;
    params.n_tris = (cl_uint)m_triangles->size();
    bvh = new BVH(m_triangles, filename);
}

void Tracer::saveHierarchy(const std::string filename)
{
    bvh->exportTo(filename);
}

void Tracer::constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode)
{
    m_triangles = &triangles;
    params.n_tris = (cl_uint)m_triangles->size();
    bvh = new BVH(m_triangles, splitMode);
}

void Tracer::initCamera()
{
    Camera cam;
    cam.pos = float3(0.0f, 1.0f, 3.5f);
    cam.right = float3(1.0f, 0.0f, 0.0f);
    cam.up = float3(0.0f, 1.0f, 0.0f);
    cam.dir = float3(0.0f, 0.0f, -1.0f);
    cam.fov = 60.0f;

    params.camera = cam;
    cameraRotation = float2(0.0f);
    paramsUpdatePending = true;
}

void Tracer::initAreaLight()
{
    params.areaLight.E = float3(1.0f, 1.0f, 1.0f) * 200.0f;
    params.areaLight.right = float3(0.0f, 0.0f, -1.0f);
    params.areaLight.up = float3(0.0f, 1.0f, 0.0f);
    params.areaLight.N = float4(-1.0f, 0.0f, 0.0f, 0.0f);
    params.areaLight.pos = float4(1.0f, 1.0f, 0.0f, 1.0f);
    params.areaLight.size = float2(0.5f, 0.5f);
    paramsUpdatePending = true;
}

// "The rows of R represent the coordinates in the original space of unit vectors along the
//  coordinate axes of the rotated space." (https://www.fastgraph.com/makegames/3drotation/)
void Tracer::updateCamera()
{
    if(cameraRotation.x < 0) cameraRotation.x += 360.0f;
    if(cameraRotation.y < 0) cameraRotation.y += 360.0f;
    if(cameraRotation.x > 360.0f) cameraRotation.x -= 360.0f;
    if(cameraRotation.y > 360.0f) cameraRotation.y -= 360.0f;

    matrix rot = rotation(float3(1, 0, 0), toRad(cameraRotation.y)) * rotation(float3(0, 1, 0), toRad(cameraRotation.x));

    params.camera.right = float3(rot.m00, rot.m01, rot.m02);
    params.camera.up =    float3(rot.m10, rot.m11, rot.m12);
    params.camera.dir =  -float3(rot.m20, rot.m21, rot.m22); // camera points in the negative z-direction
}

void Tracer::updateAreaLight()
{
    params.areaLight.right = params.camera.right;
    params.areaLight.up = params.camera.up;
    params.areaLight.N = params.camera.dir;
    params.areaLight.pos = params.camera.pos - 0.01f * params.camera.dir;
}

// Functional keys that need to be triggered only once per press
#define match(key, expr) case key: expr; paramsUpdatePending = true; break;
void Tracer::handleKeypress(int key)
{
    switch (key)
    {
        match(GLFW_KEY_M, init(params.width, params.height));
        match(GLFW_KEY_H, params.flashlight = !params.flashlight);
        match(GLFW_KEY_7, useMK = !useMK);
        match(GLFW_KEY_F1, initCamera());
        match(GLFW_KEY_F2, saveState());
        match(GLFW_KEY_F3, loadState());
        match(GLFW_KEY_SPACE, updateAreaLight());
        match(GLFW_KEY_I, std::cout << "MAX_BOUNCES: " << ++params.maxBounces << std::endl);
        match(GLFW_KEY_K, std::cout << "MAX_BOUNCES: " << (params.maxBounces > 0 ? (--params.maxBounces) : 0) << std::endl);
    }
}
#undef match

// Instant and simultaneous key presses (movement etc.)
#define check(key, expr) if(window->keyPressed(key)) { expr; paramsUpdatePending = true; }
void Tracer::pollKeys()
{
    Camera &cam = params.camera;

    check(GLFW_KEY_W,           cam.pos += cameraSpeed * 0.07f * cam.dir);
    check(GLFW_KEY_A,           cam.pos -= cameraSpeed * 0.07f * cam.right);
    check(GLFW_KEY_S,           cam.pos -= cameraSpeed * 0.07f * cam.dir);
    check(GLFW_KEY_D,           cam.pos += cameraSpeed * 0.07f * cam.right);
    check(GLFW_KEY_R,           cam.pos += cameraSpeed * 0.07f * cam.up);
    check(GLFW_KEY_F,           cam.pos -= cameraSpeed * 0.07f * cam.up);
    check(GLFW_KEY_UP,          cameraRotation.y -= 1.0f);
    check(GLFW_KEY_DOWN,        cameraRotation.y += 1.0f);
    check(GLFW_KEY_LEFT,        cameraRotation.x -= 1.0f);
    check(GLFW_KEY_RIGHT,       cameraRotation.x += 1.0f);
    check(GLFW_KEY_PERIOD,      cam.fov = std::min(cam.fov + 1.0f, 175.0f));
    check(GLFW_KEY_SEMICOLON,   cam.fov = std::max(cam.fov - 1.0f, 5.0f));
    check(GLFW_KEY_KP_ADD,      cameraSpeed += 0.1f);
    check(GLFW_KEY_0,           cameraSpeed *= 1.1f);
    check(GLFW_KEY_KP_SUBTRACT, cameraSpeed = std::max(0.05f, cameraSpeed - 0.05f));
    check(GLFW_KEY_8,           params.areaLight.size /= 1.1f);
    check(GLFW_KEY_9,           params.areaLight.size *= 1.1f);

    if(paramsUpdatePending)
    {
        updateCamera();
    }
}
#undef check

void Tracer::handleMouseButton(int key, int action)
{
    switch(key)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            if(action == GLFW_PRESS)
            {
                lastCursorPos = window->getCursorPos();
                mouseButtonState[0] = true;
                //std::cout << "Left mouse button pressed" << std::endl;
            }
            if(action == GLFW_RELEASE)
            {
                mouseButtonState[0] = false;
                //std::cout << "Left mouse button released" << std::endl;
            }
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            if(action == GLFW_PRESS) mouseButtonState[1] = true;
            if(action == GLFW_RELEASE) mouseButtonState[2] = false;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            if(action == GLFW_PRESS) mouseButtonState[2] = true;
            if(action == GLFW_RELEASE) mouseButtonState[2] = false;
            break;
    }
}

void Tracer::handleCursorPos(double x, double y)
{
    if(mouseButtonState[0])
    {
        float2 newPos = float2((float)x, (float)y);
        float2 delta =  newPos - lastCursorPos;

        // std::cout << "Mouse delta: " << delta.x <<  ", " << delta.y << std::endl;

        cameraRotation += delta;
        lastCursorPos = newPos;

        updateCamera();
        paramsUpdatePending = true;
    }
}