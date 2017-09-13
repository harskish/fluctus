#pragma once

#include <GL/glew.h>
#include "clcontext.hpp"
#include <string>
#include <cmath>
#include "geom.h"
#include "window.hpp"
#include "math/float2.hpp"
#include "math/float3.hpp"
#include "math/matrix.hpp"
#include "bvh.hpp"
#include "sbvh.hpp"
#include "scene.hpp"
#include "tinyfiledialogs.h"

using namespace FireRays;

class Tracer
{
public:
    Tracer(int width, int height);
    ~Tracer();

    bool running();
    void update();
    void resizeBuffers();
    void handleMouseButton(int key, int action);
    void handleCursorPos(double x, double y);
	void handleMouseScroll(double yoffset);
    void handleKeypress(int key); // function keys

private:
    // Create/load/export BVH
    void initHierarchy();
    void loadHierarchy(const std::string filename, std::vector<RTTriangle> &triangles);
    void saveHierarchy(const std::string filename);
    void constructHierarchy(std::vector<RTTriangle>& triangles, SplitMode splitMode);

    void pollKeys();              // movement keys
    void updateCamera();
    void updateAreaLight();
    void initCamera();
    void initAreaLight();
    void saveImage();
	
	void saveState();
    void loadState();
	enum StateIO { READ, WRITE };
	void iterateStateItems(StateIO mode);

    void selectScene(std::string file);
    void quickLoadScene(unsigned int num);
    void toggleSamplingMode();
    void toggleLightSourceMode();
    void initEnvMap();
    void init(int width, int height, std::string sceneFile = "");

    PTWindow *window;
    CLContext *clctx;
    RenderParams params;    // copied into GPU memory
    float2 cameraRotation;  // not passed to GPU but needed for camera basis vectors
    float2 lastCursorPos;
    float cameraSpeed = 1.0f;
    bool mouseButtonState[3] = { false, false, false };
    bool paramsUpdatePending = true; // force initial param update

    Scene *scene;
    BVH *bvh;
    std::vector<RTTriangle>* m_triangles;
    std::string sceneHash;
    cl_uint iteration;
    int frontBuffer = 0;
    bool hasEnvMap = false;

    bool useMK;
};
