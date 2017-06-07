#include "tracer.hpp"
#include "geom.h"

inline void printStats(const RenderStats &stats, CLContext *ctx)
{
    static double lastPrinted = 0;
    double now = glfwGetTime();
    double delta = now - lastPrinted;
    if (delta > 1.0)
    {
        lastPrinted = now;
        double scale = 1e6 * delta;
        double prim = stats.primaryRays / scale;
        double ext = stats.extensionRays / scale;
        double shdw = stats.shadowRays / scale;
        double samp = stats.samples / scale;
        printf("%.1fM primary, %.2fM extension, %.2fM shadow, %.2fM samples, total: %.2fMRays/s\r", prim, ext, shdw, samp, prim + ext + shdw);

        // Reset stat counters (synchronously...)
        ctx->resetStats();
    }
}

void Tracer::update()
{
    // React to key presses
    glfwPollEvents();
    pollKeys();

    glFinish(); // locks execution to refresh rate of display (GL)

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
        if (iteration == 0)
        {
            // Interactive preview: 1 bounce indirect
            clctx->enqueueResetKernel(params);
            clctx->enqueueRayGenKernel(params);
            
            // Two segments
            clctx->enqueueNextVertexKernel(params);
            clctx->enqueueExplSampleKernel(params);
            clctx->enqueueNextVertexKernel(params);
            clctx->enqueueExplSampleKernel(params);
            
            // Preview => also splat incomplete paths
            clctx->enqueueSplatPreviewKernel(params);
        }
        else
        {
            // Generate new camera rays
            clctx->enqueueRayGenKernel(params);

            // Trace rays
            clctx->enqueueNextVertexKernel(params);

            // Direct lighting
            clctx->enqueueExplSampleKernel(params);

            // Splat results
            clctx->enqueueSplatKernel(params, frontBuffer);
        }
    }
    else
    {
        // Megakernel
        clctx->enqueueMegaKernel(params, frontBuffer, iteration);
    }

    // Finish command queue
    clctx->finishQueue();

    // Display render statistics (MRays/s) of previous frame
    // Asynchronously transfer render statistics from device
    printStats(clctx->getStats(), clctx);
    clctx->fetchStatsAsync();

    // Draw progress to screen
    if (useMK)
    {
        window->drawPixelBuffer();
    }
    else
    {
        window->drawTexture(frontBuffer);
        frontBuffer = 1 - frontBuffer;
    }

    // Update iteration counter
    iteration++;
}

Tracer::Tracer(int width, int height) : useMK(true)
{
    // done only once (VS debugging stops working if context is recreated)
    window = new PTWindow(width, height, this); // this = glfw user pointer
    window->setShowFPS(true);
    clctx = new CLContext(window->getTexPtr(), window->getPBO());
    initCamera();
    initAreaLight();

    // done whenever a new scene is selected
    init(width, height);
}

// Run whenever a scene is laoded
void Tracer::init(int width, int height, std::string sceneFile)
{
    float renderScale = Settings::getInstance().getRenderScale();

    params.width = static_cast<unsigned int>(width * renderScale);
    params.height = static_cast<unsigned int>(height * renderScale);
    params.n_lights = sizeof(test_lights) / sizeof(PointLight);
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);
    params.useEnvMap = 0;
    params.flashlight = 0;
    params.maxBounces = 4;
    params.sampleImpl = true;
    params.sampleExpl = true;

    selectScene(sceneFile);
    loadState();
    initEnvMap();
    initHierarchy();

    clctx->uploadSceneData(bvh, scene);

    // Data uploaded to GPU => no longer needed
    delete scene;
    delete bvh;
}

// Empty file name means scene selector is opened
void Tracer::selectScene(std::string file)
{
    if (file == "")
    {
        char const * pattern[] = { "*.obj", "*.ply" };
        char const *selected = tinyfd_openFileDialog("Select a scene file", "assets/", 2, pattern, NULL, 0); // allow only single selection
        file = (selected) ? std::string(selected) : "assets/teapot.ply";
    }

    scene = new Scene(file);
    sceneHash = scene->hashString();
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
    std::string hashFile = "data/hierarchies/hierarchy_" + sceneHash + ".bin" ;
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
    window->createPBO();
    clctx->setupPixelStorage(window->getTexPtr(), window->getPBO());
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
    std::ofstream out("data/states/state_" + sceneHash + ".dat", std::ios::binary);

    // Write camera state to file
    if (out.good())
    {
        write(out, cameraRotation.x);
        write(out, cameraRotation.y);
        write(out, cameraSpeed);
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
    std::ifstream in("data/states/state_" + sceneHash + ".dat");
    if (in.good())
    {
        read(in, cameraRotation.x);
        read(in, cameraRotation.y);
        read(in, cameraSpeed);
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

// Load a scene with keys 1-5 based on shortcuts in settings.json
void Tracer::quickLoadScene(unsigned int key)
{
    auto mapping = Settings::getInstance().getShortcuts();
    auto it = mapping.find(key);
    if (it != mapping.end()) init(params.width, params.height, it->second);
}

// Controls the way light sources are sampled in path tracing
void Tracer::toggleSamplingMode()
{
    if (params.sampleImpl && params.sampleExpl) // both => expl
    {
        params.sampleImpl = false;
        std::cout << std::endl << "Sampling mode: explicit" << std::endl;
    }
    else if (params.sampleExpl) // expl => impl
    {
        params.sampleExpl = false;
        params.sampleImpl = true;
        std::cout << std::endl << "Sampling mode: implicit" << std::endl;
    }
    else // impl => both
    {
        params.sampleExpl = true;
        std::cout << std::endl << "Sampling mode: MIS" << std::endl;
    }
}

// Functional keys that need to be triggered only once per press
#define match(key, expr) case key: expr; paramsUpdatePending = true; break;
void Tracer::handleKeypress(int key)
{
    switch (key)
    {
        match(GLFW_KEY_1,           quickLoadScene(1));
        match(GLFW_KEY_2,           quickLoadScene(2));
        match(GLFW_KEY_3,           quickLoadScene(3));
        match(GLFW_KEY_4,           quickLoadScene(4));
        match(GLFW_KEY_5,           quickLoadScene(5));
        match(GLFW_KEY_L,           init(params.width, params.height));  // opens scene selector
        match(GLFW_KEY_H,           params.flashlight = !params.flashlight);
        match(GLFW_KEY_7,           useMK = !useMK);
        match(GLFW_KEY_F1,          initCamera());
        match(GLFW_KEY_F2,          saveState());
        match(GLFW_KEY_F3,          loadState());
        match(GLFW_KEY_SPACE,       updateAreaLight());
        match(GLFW_KEY_I,           std::cout << std::endl << "MAX_BOUNCES: " << ++params.maxBounces << std::endl);
        match(GLFW_KEY_K,           std::cout << std::endl << "MAX_BOUNCES: " << (params.maxBounces > 0 ? (--params.maxBounces) : 0) << std::endl);
        match(GLFW_KEY_M,           toggleSamplingMode());
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
    check(GLFW_KEY_8,           params.areaLight.size /= 1.1f);
    check(GLFW_KEY_9,           params.areaLight.size *= 1.1f);
    check(GLFW_KEY_PAGE_DOWN,   params.areaLight.E /= 1.05f);
    check(GLFW_KEY_PAGE_UP,     params.areaLight.E *= 1.05f);

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
            if(action == GLFW_RELEASE) mouseButtonState[1] = false;
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

void Tracer::handleMouseScroll(double yoffset)
{
    float newSpeed = (yoffset > 0) ? cameraSpeed * 1.2f : cameraSpeed / 1.2f;
    cameraSpeed = std::max(1e-3f, std::min(1e6f, newSpeed));
}