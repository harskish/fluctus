#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>
#include <string>
#include <map>
#include "sbvh.hpp"
#include "scene.hpp"
#include "math/float2.hpp"
#include "math/float3.hpp"
#include "math/matrix.hpp"
#include "geom.h"

#include "HWAccelerator.hpp"

#ifdef WITH_OPTIX
#include "OptixDenoiser.hpp"
#endif

using namespace FireRays;

class CLContext;
class PTWindow;
class BVH;
class Scene;

struct FloatWidget;

class Tracer
{
public:
    Tracer(int width, int height);
    ~Tracer();

    // Load given scene (or open selector)
    void init(int width, int height, std::string sceneFile = "");

    // Two modes of operation
    void renderInteractive();
    void renderSingle(int spp, bool denoise = false);

    bool running();
    void update();
    void runBenchmark();
    void resizeBuffers(int w, int h);
    void handleMouseButton(int key, int action, int mods);
    void handleCursorPos(double x, double y);
	void handleMouseScroll(double yoffset);
    void handleKeypress(int key, int scancode, int action, int mods); // function keys
    void handleChar(unsigned int codepoint);
    void handleFileDrop(int count, const char **filenames);

    CLContext* getClContext() { return clctx; };
    const RenderParams& getParams() { return params; }
    std::shared_ptr<Scene> getScene() { return scene; }

    bool useDenoiser = false;

    // GUI - implemented in tracer_ui.cpp
private:
    FloatWidget* addFloatWidget(nanogui::Popup* parent, std::string title,
        std::string key, float vMin, float vMax, std::function<void(float)> updateFunc);
    void setupToolbar();
    void resetParams(int width, int height);
    void addRendererSettings(nanogui::Widget *parent);
    void addCameraSettings(nanogui::Widget *parent);
    void addTonemapSettings(nanogui::Widget *parent);
    void addDenoiserSettings(nanogui::Widget *parent);
    void addEnvMapSettings(nanogui::Widget *parent);
    void addAreaLightSettings(nanogui::Widget *parent);
    void addStateSettings(nanogui::Widget *parent);
    void updateGUI();
    void toggleGUI();
    bool shouldSkipPoll();
    nanogui::Window *tools;
    std::map<std::string, void*> uiMapping;
    std::vector<nanogui::TextBox*> inputBoxes; // for checking focus

private:
    // Create/load/export BVH
    void initHierarchy();
    void loadHierarchy(const std::string filename, std::vector<RTTriangle> &triangles);
    void saveHierarchy(const std::string filename);
    void constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode, ProgressView* progress);

    void pollKeys(float deltaT); // movement keys
    void updateCamera();
    void updateAreaLight();
    void initCamera();
    void initPostProcessing();
    void initAreaLight();
    void saveImage();

    // Shoot single picking ray through cursor
    Hit pickSingle();

    // Set DoF depth by right clicking
    void pickDofDepth();
	
	void saveState();
    void loadState();
	enum StateIO { READ, WRITE };
	void iterateStateItems(StateIO mode);

    void selectScene(std::string file);
    void quickLoadScene(unsigned int num);
    void toggleSamplingMode();
    void toggleLightSourceMode();
    void toggleRenderer();
    void toggleDenoiserVisibility();
    void initEnvMap();

#ifdef WITH_OPTIX
    OptixDenoiser denoiser;
    optix::Buffer denoisedResult;
    float denoiserStrength = 1.0f;
#endif

    PTWindow *window;
    CLContext *clctx;
    RenderParams params;    // copied into GPU memory
    float2 cameraRotation;  // not passed to GPU but needed for camera basis vectors
    float2 lastCursorPos;
    double lastUpdate = 0.0f;
    float cameraSpeed = 1.0f;
    bool mouseButtonState[3] = { false, false, false };
    bool paramsUpdatePending = true; // force initial param update

    std::shared_ptr<Scene> scene;
    std::shared_ptr<EnvironmentMap> envMap;
    BVH *bvh = nullptr;
    std::vector<RTTriangle>* m_triangles;
    std::string sceneHash;
    cl_uint iteration;
    int frontBuffer = 0;
    bool hasEnvMap = false;

    bool useWavefront;

    HWAccelerator accelerator;
};
