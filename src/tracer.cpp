#include "tracer.hpp"
#include "geom.h"
#include "progressview.hpp"

Tracer::Tracer(int width, int height) : useWavefront(true)
{
    // Before UI
    initCamera();
    initPostProcessing();
    initAreaLight();

    // done only once (VS debugging stops working if context is recreated)
    window = new PTWindow(width, height, this); // this = glfw user pointer
    window->setShowFPS(true);
    clctx = new CLContext();
    window->setCLContextPtr(clctx);
    window->setupGUI();
    clctx->setup(window);
    setupToolbar();

    // done whenever a new scene is selected
    init(width, height);
    
    // Show toolbar
    toggleGUI();
}

// Run whenever a scene is loaded
void Tracer::init(int width, int height, std::string sceneFile)
{
    float renderScale = Settings::getInstance().getRenderScale();

    params.width = static_cast<unsigned int>(width * renderScale);
    params.height = static_cast<unsigned int>(height * renderScale);
    params.n_lights = sizeof(test_lights) / sizeof(PointLight);
    params.useEnvMap = (cl_uint)false;
    params.useAreaLight = (cl_uint)true;
    params.envMapStrength = 1.0f;
    params.flashlight = (cl_uint)false;
    params.maxBounces = 6;
    params.sampleImpl = (cl_uint)true;
    params.sampleExpl = (cl_uint)true;
    params.useRoulette = (cl_uint)false;

    window->showMessage("Loading scene");
    selectScene(sceneFile);
    loadState();
    window->showMessage("Creating BVH");
    initHierarchy();

    // Diagonal gives maximum ray length within the scene
    AABB_t bounds = bvh->getSceneBounds();
    params.worldRadius = (cl_float)(length(bounds.max - bounds.min) * 0.5f);

    window->showMessage("Uploading scene data");
    clctx->uploadSceneData(bvh, scene.get());

    // Data uploaded to GPU => no longer needed
    delete bvh;

    // Setup GUI sliders with correct values
    updateGUI();

    // Hide status message
    window->hideMessage();
}

inline void printStats(CLContext *ctx)
{
    static double lastPrinted = 0;
    double now = glfwGetTime();
    double delta = now - lastPrinted;
    if (delta > 1.0)
    {
        lastPrinted = now;
        ctx->updateRenderPerf(delta); // updated perf can now be accessed from anywhere
        PerfNumbers perf = ctx->getRenderPerf();
        printf("%.1fM primary, %.1fM extension, %.1fM shadow, %.1fM samples, total: %.1fMRays/s\r",
            perf.primary, perf.extension, perf.shadow, perf.samples, perf.total);

        // Reset stat counters (synchronously...)
        ctx->resetStats();
    }
}

void Tracer::update()
{
    // Calculate time since last update
    double newT = glfwGetTime();
    float deltaT = (float)std::min(newT - lastUpdate, 0.1); // seconds
    lastUpdate = newT;
    
    // React to key presses
    glfwPollEvents();
    pollKeys(deltaT);

    glFinish(); // locks execution to refresh rate of display (GL)

    // Update RenderParams in GPU memory if needed
    if(paramsUpdatePending)
    {
        // Update render dimensions
        const float renderScale = Settings::getInstance().getRenderScale();
        window->getFBSize(params.width, params.height);
        params.width = static_cast<unsigned int>(params.width * renderScale);
        params.height = static_cast<unsigned int>(params.height * renderScale);

        updateGUI();
        clctx->updateParams(params);
        paramsUpdatePending = false;
        iteration = 0; // accumulation reset
    }

    QueueCounters cnt;
    
    if (useWavefront)
    {
        // Aila-style WF
        cl_uint maxBounces = params.maxBounces;
        int N = 1;
        
        if (iteration == 0)
        {
            // Set to 2-bounce for preview
            params.maxBounces = std::min((cl_uint)2, maxBounces);
            clctx->updateParams(params);
            N = 3;

            // Create and trace primary rays
            clctx->resetPixelIndex();
            clctx->enqueueWfResetKernel(params);
            clctx->enqueueWfLogicKernel(params);
            clctx->enqueueWfRaygenKernel(params);
            clctx->enqueueWfExtRayKernel(params);
            clctx->enqueueClearWfQueues();
        }

        // Advance wavefront N segments
        for (int i = 0; i < N; i++)
        {
            // Fill queues
            clctx->enqueueWfLogicKernel(params);

            // Operate on queues
            clctx->enqueueWfRaygenKernel(params);
            clctx->enqueueWfMaterialKernels(params);
            clctx->enqueueGetCounters(&cnt); // the subsequent kernels don't grow the queues
            clctx->enqueueWfExtRayKernel(params);
            clctx->enqueueWfShadowRayKernel(params);

            // Clear queues
            clctx->enqueueClearWfQueues();
        }

        // Postprocess
        clctx->enqueuePostprocessKernel(params);

        // Reset bounces
        if (iteration == 0)
        {
            params.maxBounces = maxBounces;
            clctx->updateParams(params);
        }
    }
    else
    {
        // Luxrender-style microkernels
        if (iteration == 0)
        {
            // Interactive preview: 1 bounce indirect
            clctx->enqueueResetKernel(params);
            clctx->enqueueRayGenKernel(params);

            // Two segments
            clctx->enqueueNextVertexKernel(params);
            clctx->enqueueBsdfSampleKernel(params, iteration);
            clctx->enqueueNextVertexKernel(params);
            clctx->enqueueBsdfSampleKernel(params, iteration + 1);

            // Preview => also splat incomplete paths
            clctx->enqueueSplatPreviewKernel(params);
        }
        else
        {
            // Generate new camera rays
            clctx->enqueueRayGenKernel(params);

            // Trace rays
            clctx->enqueueNextVertexKernel(params);

            // Direct lighting + environment map IS
            clctx->enqueueBsdfSampleKernel(params, iteration);

            // Splat results
            clctx->enqueueSplatKernel(params, iteration);
        }

        // Postprocess
        clctx->enqueuePostprocessKernel(params);
    }

    // Finish command queue
    clctx->finishQueue();

    // Enqueue WF pixel index update
    clctx->updatePixelIndex(params.width * params.height, cnt.raygenQueue);

    // Draw progress to screen
    window->draw();

    
    if (useWavefront)
    {
        // Update statsAsync based on queues
        clctx->statsAsync.extensionRays += cnt.extensionQueue;
        clctx->statsAsync.shadowRays += cnt.shadowQueue;
        clctx->statsAsync.primaryRays += cnt.raygenQueue;
        clctx->statsAsync.samples += (iteration > 0) ? cnt.raygenQueue : 0;
    }
    else
    {
        // Explicit atomic render stats only on MK
        clctx->fetchStatsAsync();
    }

    // Display render statistics (MRays/s)
    printStats(clctx);

    // Update iteration counter
    iteration++;

    if (iteration % 1000 == 0)
        saveImage();
}

// Runs benchmark on conference, egyptcat and kitchen (30s each)
// Generates csv (stats over time) or txt (averages)
void Tracer::runBenchmark()
{
    // Setup renderer state for benchmarking
    params.width = 1024;
    params.height = 1024;
    Settings::getInstance().setRenderScale(1.0f);
    window->setSize(params.width, params.height);
    updateGUI();

    // Called when scene changes
    auto resetRenderer = [&]()
    {
        iteration = 0;   
        glFinish();
        clctx->updateParams(params);
        clctx->enqueueResetKernel(params);
        clctx->enqueueWfResetKernel(params);
        clctx->finishQueue();
        clctx->resetStats();
    };
    
    std::vector<const char*> scenes =
    { 
        "assets/egyptcat/egyptcat.obj",
        "assets/conference/conference.obj",
        "assets/country_kitchen/Country-Kitchen.obj",
    };

    std::stringstream simpleReport;
    std::stringstream csvReport;
    csvReport << "scene;time;primary;extension;shadow;total;samples\n";

    // Stats include time dimension
    std::vector<RenderStats> statsLog;
    double lastLogTime = 0;

    auto logStats = [&](const char* scene, double elapsed, double deltaT)
    {
        RenderStats stats = clctx->getStats();
        statsLog.push_back(stats);
        clctx->resetStats();
        lastLogTime = glfwGetTime();
        double s = 1e6 * deltaT;
        csvReport << scene << ";" << elapsed << ";" << stats.primaryRays / s << ";"
            << stats.extensionRays / s << ";" << stats.shadowRays / s << ";"
            << (stats.primaryRays + stats.extensionRays + stats.shadowRays) / s << ";"
            << stats.samples / s << "\n";
    };

    toggleGUI();
    window->setShowFPS(false);
    auto prg = window->getProgressView();
    
    const double RENDER_LEN = 30.0;
    for (int i = 0; i < scenes.size(); i++) {
        std::string counter = std::to_string(i + 1) + "/" + std::to_string(scenes.size());
        init(params.width, params.height, scenes[i]);
        resetRenderer();

        double startT = glfwGetTime();
        double currT = startT;
        while (currT - startT < RENDER_LEN)
        {
            QueueCounters cnt;

            glfwPollEvents();
            if (!window->available()) exit(0); // react to exit button

            if (useWavefront)
            {
                clctx->enqueueWfLogicKernel(params);
                clctx->enqueueWfRaygenKernel(params);
                clctx->enqueueWfMaterialKernels(params);
                clctx->enqueueGetCounters(&cnt);
                clctx->enqueueWfExtRayKernel(params);
                clctx->enqueueWfShadowRayKernel(params);
                clctx->enqueueClearWfQueues();
            }
            else
            {
                clctx->enqueueRayGenKernel(params);
                clctx->enqueueNextVertexKernel(params);
                clctx->enqueueBsdfSampleKernel(params, iteration);
                clctx->enqueueSplatKernel(params, frontBuffer);
            }

            clctx->enqueuePostprocessKernel(params);

            // Synchronize
            clctx->finishQueue();

            // Update statistics
            if (useWavefront)
            {
                // Update statsAsync based on queues
                clctx->statsAsync.extensionRays += cnt.extensionQueue;
                clctx->statsAsync.shadowRays += cnt.shadowQueue;
                clctx->statsAsync.primaryRays += cnt.raygenQueue;
                clctx->statsAsync.samples += (iteration > 0) ? cnt.raygenQueue : 0;
            }
            else
            {
                // Fetch explicit stats from device
                clctx->fetchStatsAsync();
            }

            // Update index of next pixel to shade
            clctx->updatePixelIndex(params.width * params.height, cnt.raygenQueue);

            // Draw image + loading bar
            prg->showMessage("Running benchmark " + counter, (currT - startT) / RENDER_LEN);

            // Save statistics every half a second to log for further processing
            double deltaT = currT - lastLogTime;
            if (deltaT > 0.5)
                logStats(scenes[i], currT - startT, deltaT);

            iteration++;
            currT = glfwGetTime();
        }

        // Process statistics for current scene
        logStats(scenes[scenes.size() - 1], currT - startT, currT - lastLogTime);
        double time = currT - startT;
        unsigned long long sums[] = { 0, 0, 0, 0 };
        for (RenderStats &s : statsLog)
        {
            sums[0] += s.primaryRays;
            sums[1] += s.extensionRays;
            sums[2] += s.shadowRays;
            sums[3] += s.samples;
        }

        double scale = 1e6 * time;
        double prim = sums[0] / scale;
        double ext = sums[1] / scale;
        double shdw = sums[2] / scale;
        double samp = sums[3] / scale;
        
        char statistics[512];
        sprintf(statistics, "%s: %.1fM primary, %.2fM extension, %.2fM shadow, %.2fM samples, total: %.2fM rays/s", scenes[i], prim, ext, shdw, samp, prim + ext + shdw);
        std::cout << statistics << std::endl;
        simpleReport << statistics << std::endl;
        statsLog.clear();
    }

    prg->hide();
    toggleGUI();
    window->setShowFPS(true);

    // Output report
    std::string outpath = saveFileDialog("Save results", "", { "*.txt", "*.csv" });
    if (outpath != "")
    {
        if (!endsWith(outpath, ".csv") && !endsWith(outpath, ".txt")) outpath += ".txt";
        std::ofstream outfile(outpath);

        if (!outfile.good()) {
            std::cout << "Failed to write benchmark report!" << std::endl;
            return;
        }

        std::string contents = (endsWith(outpath, ".csv")) ? csvReport.str() : simpleReport.str();
        outfile << contents;
    }
}

// Empty file name means scene selector is opened
void Tracer::selectScene(std::string file)
{
    if (file == "")
    {
        std::string selected = openFileDialog("Select a scene file", "assets/", { "*.obj", "*.ply" });
        file = (selected != "") ? selected : "assets/egyptcat/egyptcat.obj";
    }

    scene.reset(new Scene());
    scene->loadModel(file, window->getProgressView());
    if (envMap)
        scene->setEnvMap(envMap);

    sceneHash = scene->hashString();

    std::string envMapName = Settings::getInstance().getEnvMapName();
    if (envMapName == "")
        return;

    if (!envMap || envMap->getName() != envMapName)
    {
        envMap.reset(new EnvironmentMap(envMapName));
        scene->setEnvMap(envMap);
        initEnvMap();
    }
    else
    {
        std::cout << "Reusing environment map" << std::endl;
    }
}

void Tracer::initEnvMap()
{
    // Bool operator => check if ptr is empty
    if (envMap && envMap->valid())
    {
        params.useEnvMap = (cl_int)true;
        this->hasEnvMap = true;
        clctx->createEnvMap(envMap.get());
    }
}

// Check if old hierarchy can be reused
void Tracer::initHierarchy()
{
	std::string hashFile = "data/hierarchies/hierarchy_" + sceneHash + ".bin";
    std::ifstream input(hashFile, std::ios::in);

    if (input.good())
    {
        std::cout << "Reusing BVH..." << std::endl;
        loadHierarchy(hashFile, scene->getTriangles());
    }
    else
    {
        std::cout << "Building BVH..." << std::endl;
        constructHierarchy(scene->getTriangles(), SplitMode_Sah, window->getProgressView());
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
void Tracer::resizeBuffers(int width, int height)
{
    window->getScreen()->resizeCallbackEvent(width, height);
    ProgressView *pv = window->getProgressView();
    if (pv) pv->center();

    window->createTextures();
    window->createPBO();
    clctx->setupPixelStorage(window);
    paramsUpdatePending = true;
}

inline void writeVec(std::fstream &out, FireRays::float3 &vec)
{
    write(out, vec.x);
    write(out, vec.y);
    write(out, vec.z);
}

inline void readVec(std::fstream &in, FireRays::float3 &vec)
{
	read(in, vec.x);
	read(in, vec.y);
	read(in, vec.z);
}

// Shared method for read/write => no forgotten members
void Tracer::iterateStateItems(StateIO mode)
{
	#define rw(item) if(mode == StateIO::WRITE) write(stream, item); else read(stream, item);
	#define rwVec(item) if(mode == StateIO::WRITE) writeVec(stream, item); else readVec(stream, item);

	auto fileMode = std::ios::binary | ((mode == StateIO::WRITE) ? std::ios::out : std::ios::in);
	std::fstream stream("data/states/state_" + sceneHash + ".dat", fileMode);

	if (stream.good())
	{
		// Camera
		rw(cameraRotation.x);
		rw(cameraRotation.y);
		rw(cameraSpeed);
		rw(params.camera.fov);
        rw(params.camera.focalDist);
        rw(params.camera.apertureSize);
		rwVec(params.camera.dir);
		rwVec(params.camera.pos);
		rwVec(params.camera.right);
		rwVec(params.camera.up);

		// Lights
		rwVec(params.areaLight.N);
		rwVec(params.areaLight.pos);
		rwVec(params.areaLight.right);
		rwVec(params.areaLight.up);
		rwVec(params.areaLight.E);
		rw(params.areaLight.size.x);
		rw(params.areaLight.size.y);
		rw(params.envMapStrength);

		// Sampling parameters
		rw(params.maxBounces);
		rw(params.useAreaLight);
		rw(params.useEnvMap);
		rw(params.sampleExpl);
		rw(params.sampleImpl);
        rw(params.useRoulette);

        // Post processing
        rw(params.ppParams.exposure);
        rw(params.ppParams.tmOperator);

		std::cout << ((mode == StateIO::WRITE) ? "State dumped" : "State imported") << std::endl;
	}
	else
	{
		std::cout << "Could not open state file" << std::endl;
	}

	#undef rw
	#undef rwVec
}

Hit Tracer::pickSingle()
{
    // Position relative to upper-left
    double xpos, ypos;
    glfwGetCursorPos(window->glfwWindowPtr(), &xpos, &ypos);

    int width, height;
    glfwGetWindowSize(window->glfwWindowPtr(), &width, &height);

    // Ignores FB scaling
    float NDCx = xpos / width;
    float NDCy = (height - ypos) / height;

    return clctx->pickSingle(NDCx, NDCy);
}

// Set DoF depth based on hit distance
void Tracer::pickDofDepth()
{
    Hit hit = pickSingle();
    
    printf("Pick result: i = %d, dist = %.2f\n\n", hit.i, hit.t);

    // If scene hit, set focal distance
    if (hit.i > -1)
    {
        params.camera.focalDist = hit.t;
        paramsUpdatePending = true;
    }
}

void Tracer::saveState()
{
	iterateStateItems(StateIO::WRITE);
}

void Tracer::loadState()
{
	iterateStateItems(StateIO::READ);
}

void Tracer::saveImage()
{
    std::time_t epoch = std::time(nullptr);
    std::string fileName = "output_" + std::to_string(epoch) + ".png";
    clctx->saveImage(fileName, params);
}

void Tracer::loadHierarchy(const std::string filename, std::vector<RTTriangle>& triangles)
{
    m_triangles = &triangles;
    params.n_tris = (cl_uint)m_triangles->size();
    bvh = new SBVH(m_triangles, filename);
}

void Tracer::saveHierarchy(const std::string filename)
{
    bvh->exportTo(filename);
}

void Tracer::constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode, ProgressView *progress)
{
    m_triangles = &triangles;
    params.n_tris = (cl_uint)m_triangles->size();
    bvh = new SBVH(m_triangles, splitMode, progress);
}

void Tracer::initCamera()
{
    Camera cam;
    cam.pos = float3(0.0f, 1.0f, 3.5f);
    cam.right = float3(1.0f, 0.0f, 0.0f);
    cam.up = float3(0.0f, 1.0f, 0.0f);
    cam.dir = float3(0.0f, 0.0f, -1.0f);
    cam.fov = 60.0f;
    cam.apertureSize = 0.0f;
    cam.focalDist = 0.5f;

    params.camera = cam;
    cameraRotation = float2(0.0f);
    cameraSpeed = 1.0f;

    paramsUpdatePending = true;
}

void Tracer::initPostProcessing()
{
    PostProcessParams p;
    p.exposure = 1.0f;
    p.tmOperator = 2; // UC2 default

    params.ppParams = p;
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

void Tracer::toggleLightSourceMode()
{
    if (!hasEnvMap)
    {
        std::cout << std::endl << "No environment map loaded!" << std::endl;
    }
    else if (params.useAreaLight && params.useEnvMap) // both => env
    {
        params.useAreaLight = false;
        std::cout << std::endl << "Light mode: environment" << std::endl;
    }
    else if (params.useEnvMap) // env => area
    {
        params.useEnvMap = false;
        params.useAreaLight = true;
        std::cout << std::endl << "Light mode: area light" << std::endl;
    }
    else // area => both
    {
        params.useEnvMap = true;
        std::cout << std::endl << "Light mode: both" << std::endl;
    }
}

void Tracer::toggleRenderer()
{
    useWavefront = !useWavefront;
    window->setRenderMethod((useWavefront) ? PTWindow::RenderMethod::WAVEFRONT : PTWindow::RenderMethod::MICROKERNEL);
}

void Tracer::handleChar(unsigned int codepoint)
{
    window->getScreen()->charCallbackEvent(codepoint);
}

void Tracer::handleFileDrop(int count, const char **filenames)
{
    if (window->getScreen()->dropCallbackEvent(count, filenames)) return;

    for (int i = 0; i < count; i++)
    {
        std::string file(filenames[i]);
        if (endsWith(file, ".obj") || endsWith(file, ".ply"))
        {
            init(params.width, params.height, file);
            paramsUpdatePending = true;
            return;
        }
        if (endsWith(file, ".hdr"))
        {
            if (!envMap || envMap->getName() != file)
            {
                Settings::getInstance().setEnvMapName(file);
                envMap.reset(new EnvironmentMap(file));
                scene->setEnvMap(envMap);
                initEnvMap();
                paramsUpdatePending = true;
            }
            
            return;
        }
    }

    std::cout << "Unknown file format" << std::endl;
}

// Functional keys that need to be triggered only once per press
#define matchInit(key, expr) case key: expr; paramsUpdatePending = true; break;
#define matchKeep(key, expr) case key: expr; break;
void Tracer::handleKeypress(int key, int scancode, int action, int mods)
{
    if (window->getScreen()->keyCallbackEvent(key, scancode, action, mods)) return;

    switch (key)
    {
        // Force init
        matchInit(GLFW_KEY_1,           quickLoadScene(1));
        matchInit(GLFW_KEY_2,           quickLoadScene(2));
        matchInit(GLFW_KEY_3,           quickLoadScene(3));
        matchInit(GLFW_KEY_4,           quickLoadScene(4));
        matchInit(GLFW_KEY_5,           quickLoadScene(5));
        matchInit(GLFW_KEY_L,           init(params.width, params.height));  // opens scene selector
        matchInit(GLFW_KEY_H,           toggleLightSourceMode());
        matchInit(GLFW_KEY_7,           toggleRenderer());
        matchInit(GLFW_KEY_F1,          initCamera());
        matchInit(GLFW_KEY_F3,          loadState());
        matchInit(GLFW_KEY_SPACE,       updateAreaLight());
        matchInit(GLFW_KEY_I,           params.maxBounces += 1);
        matchInit(GLFW_KEY_K,           params.maxBounces = std::max(1u, params.maxBounces) - 1);
        matchInit(GLFW_KEY_M,           toggleSamplingMode());

        // Don't force init
        matchKeep(GLFW_KEY_F2,          saveState());
        matchKeep(GLFW_KEY_F5,          saveImage());
        matchKeep(GLFW_KEY_U,           toggleGUI());
    }
}
#undef matchInit
#undef matchKeep

// Instant and simultaneous key presses (movement etc.)
#define check(key, expr) if(window->keyPressed(key)) { expr; paramsUpdatePending = true; }
void Tracer::pollKeys(float deltaT)
{
    if (shouldSkipPoll()) return;
    
    Camera &cam = params.camera;

    check(GLFW_KEY_W,           cam.pos += deltaT * cameraSpeed * 10 * cam.dir);
    check(GLFW_KEY_A,           cam.pos -= deltaT * cameraSpeed * 10 * cam.right);
    check(GLFW_KEY_S,           cam.pos -= deltaT * cameraSpeed * 10 * cam.dir);
    check(GLFW_KEY_D,           cam.pos += deltaT * cameraSpeed * 10 * cam.right);
    check(GLFW_KEY_R,           cam.pos += deltaT * cameraSpeed * 10 * cam.up);
    check(GLFW_KEY_F,           cam.pos -= deltaT * cameraSpeed * 10 * cam.up);
    check(GLFW_KEY_UP,          cameraRotation.y -= 75 * deltaT);
    check(GLFW_KEY_DOWN,        cameraRotation.y += 75 * deltaT);
    check(GLFW_KEY_LEFT,        cameraRotation.x -= 75 * deltaT);
    check(GLFW_KEY_RIGHT,       cameraRotation.x += 75 * deltaT);
    check(GLFW_KEY_PERIOD,      cam.fov = (cl_float)std::min(cam.fov + 70 * deltaT, 175.0f));
    check(GLFW_KEY_COMMA,       cam.fov = (cl_float)std::max(cam.fov - 70 * deltaT, 5.0f));
    check(GLFW_KEY_8,           params.areaLight.size /= (cl_float)(1 + 5 * deltaT));
    check(GLFW_KEY_9,           params.areaLight.size *= (cl_float)(1 + 5 * deltaT));
    check(GLFW_KEY_PAGE_DOWN,   params.areaLight.E /= (cl_float)(1 + 10 * deltaT));
    check(GLFW_KEY_PAGE_UP,     params.areaLight.E *= (cl_float)(1 + 10 * deltaT));
    check(GLFW_KEY_X,           params.envMapStrength *= (cl_float)(1 + 5 * deltaT));
    check(GLFW_KEY_Z,           params.envMapStrength /= (cl_float)(1 + 5 * deltaT));

    if(paramsUpdatePending)
    {
        updateCamera();
    }
}
#undef check

void Tracer::handleMouseButton(int key, int action, int mods)
{
    if (window->getScreen()->mouseButtonCallbackEvent(key, action, mods)) return;

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
            if (action == GLFW_RELEASE)
            {
                mouseButtonState[2] = false;
                pickDofDepth();
            }
            break;
    }
}

void Tracer::handleCursorPos(double x, double y)
{
    if (window->getScreen()->cursorPosCallbackEvent(x, y)) return;

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
    updateGUI();
}