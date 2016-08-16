#include "tracer.hpp"
#include "geom.h"

Tracer::Tracer(int width, int height)
{
    scene = new Scene("assets/garg.obj");
	initHierarchy();

    window = new PTWindow(width, height, this);
    window->setShowFPS(true);
    clctx = new CLContext(window->getPBO());
    clctx->createBVHBuffers(bvh->m_triangles, &bvh->m_indices, &bvh->m_nodes);

    params.width = (unsigned int)width;
    params.height = (unsigned int)height;
    params.n_lights = sizeof(test_lights) / sizeof(Light);
    params.n_objects = sizeof(test_spheres) / sizeof(Sphere);

    initCamera();
	loadCameraState(); // useful when debugging
}

// Check if old hierarchy can be reused
void Tracer::initHierarchy()
{
	std::string hashFile = "hierarchy-" + scene->hashString() + ".bin" ;
	std::ifstream input(hashFile, std::ios::in);

	if (input.good())
	{
		std::cout << "Reusing BVH..." << std::endl;
		loadHierarchy(hashFile.c_str(), scene->getTriangles());
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
    delete scene;
    delete bvh;
}

bool Tracer::running()
{
    return window->available();
}

// Callback for when the window size changes
void Tracer::resizeBuffers()
{
    window->createPBO();
    clctx->createPBO(window->getPBO());
    paramsUpdatePending = true;
    std::cout << std::endl;
}

void Tracer::update()
{
    // React to key presses
    glfwPollEvents();
    pollKeys();

    // Update RenderParams in GPU memory if needed
    window->getFBSize(params.width, params.height);
    if(paramsUpdatePending)
    {
        clctx->updateParams(params);
        paramsUpdatePending = false;
    }

    // Advance render state
    clctx->executeKernel(params);

    // Draw progress to screen
    window->repaint();
}

inline void writeVec(std::ofstream &out, FireRays::float3 &vec)
{
	write(out, vec.x);
	write(out, vec.y);
	write(out, vec.z);
}

void Tracer::saveCameraState()
{
	std::ofstream out("camera.dat", std::ios::binary);

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
	}
	else
	{
		std::cout << "Could not create camera state file" << std::endl;
	}
}

inline void readVec(std::ifstream &in, FireRays::float3 &vec)
{
	read(in, vec.x);
	read(in, vec.y);
	read(in, vec.z);
}

void Tracer::loadCameraState()
{
	std::ifstream in("camera.dat");
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
	}
	else
	{
		std::cout << "Camera state file not found" << std::endl;
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

// "The rows of R represent the coordinates in the original space of unit vectors along the
// coordinate axes of the rotated space." (https://www.fastgraph.com/makegames/3drotation/)
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

// Polling enables instant and simultaneous key presses (callbacks less so)
#define check(key, expr) if(window->keyPressed(key)) { expr; paramsUpdatePending = true; }
void Tracer::pollKeys()
{
    Camera &cam = params.camera;

    check(GLFW_KEY_W,           cam.pos += 0.07f * cam.dir);
    check(GLFW_KEY_A,           cam.pos -= 0.07f * cam.right);
    check(GLFW_KEY_S,           cam.pos -= 0.07f * cam.dir);
    check(GLFW_KEY_D,           cam.pos += 0.07f * cam.right);
    check(GLFW_KEY_R,           cam.pos += 0.07f * cam.up);
    check(GLFW_KEY_F,           cam.pos -= 0.07f * cam.up);
    check(GLFW_KEY_KP_ADD,      cam.fov = std::min(cam.fov + 1.0f, 175.0f));
    check(GLFW_KEY_KP_SUBTRACT, cam.fov = std::max(cam.fov - 1.0f, 5.0f));
    check(GLFW_KEY_UP,          cameraRotation.y -= 1.0f);
    check(GLFW_KEY_DOWN,        cameraRotation.y += 1.0f);
    check(GLFW_KEY_LEFT,        cameraRotation.x -= 1.0f);
    check(GLFW_KEY_RIGHT,       cameraRotation.x += 1.0f);
    check(GLFW_KEY_F1,          initCamera());
	check(GLFW_KEY_F2,			saveCameraState());
	check(GLFW_KEY_F3,          loadCameraState());
    check(GLFW_KEY_ESCAPE,      window->requestClose());

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