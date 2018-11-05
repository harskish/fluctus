#include "clcontext.hpp"
#include "utils.h"
#include "geom.h"
#include "triangle.hpp"
#include "bvhnode.hpp"
#include "settings.hpp"
#include "bvh.hpp"
#include "scene.hpp"
#include "texture.hpp"
#include "window.hpp"
#include "kernel_impl.hpp"
#include "IL/ilu.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h> // texture conversion stuff
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/OpenGL.h>
#elif defined(__linux__)
#include <GL/glxew.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#endif

CLContext::CLContext()
{
    printDevices();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    platform = getPlatformByName(platforms, Settings::getInstance().getPlatformName());
    std::cout << "PLATFORM: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	#ifdef CPU_DEBUGGING
		platform.getDevices(CL_DEVICE_TYPE_CPU, &clDevices);
	#else
		platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);
	#endif

    // Init shared context
    #ifdef __APPLE__
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)kCGLShareGroup, 0
        };
    #else
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
        #if defined(__linux__)
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        #elif defined(_WIN32)
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        #endif
            0
        };
    #endif

    // Select correct device from context based on settings
    device = getDeviceByName(clDevices, Settings::getInstance().getDeviceName());
    std::cout << "DEVICE: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // Restrict context to selected device
    clDevices = { device };
    context = cl::Context(clDevices, props, NULL, NULL, &err);
    verify("Failed to create shared context");

    // Create command queue for context
    cmdQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    verify("Failed to create command queue!");

    // Setup WF task buffer size
    cl_uint bufferSize = Settings::getInstance().getWfBufferSize();
    NUM_TASKS = bufferSize;
}

void CLContext::setup(PTWindow *window)
{
    this->window = window;

    // Set global OpenCL build settings
    setKernelBuildSettings();

    // Setup RenderParams
    setupParams();

    // Setup pick result buffer
    setupPickResult();

    // Setup RenerStats
    setupStats();

    // Create OpenCL buffer from OpenGL PBO
    setupPixelStorage(window);

    // Allocate device memory for scene
    setupScene();

    // Build kernels, set their params
    initMCBuffers();
    
    // Kernels are setup after loading a scene
    // This is so that only the relevant material code is included
}

void CLContext::setupKernels()
{
    // Microkernels
	setupResetKernel();
    setupRayGenKernel();
    setupNextVertexKernel();
    setupBsdfSampleKernel();
    setupSplatKernel();
    setupSplatPreviewKernel();

    // Wavefront kernels
    setupWfResetKernel();
    setupWfExtKernel();
    setupWfRaygenKernel();
    setupWfLogicKernel();
    setupWfShadowKernel();
    setupWfDiffuseKernel();
    setupWfGlossyKernel();
    setupWfGGXReflKernel();
    setupWfGGXRefrKernel();
    setupWfDeltaKernel();
    setupWfAllMaterialsKernel();

    // Other
    setupPickKernel();
    setupPostprocessKernel();
}

// For copying SoA data to host
inline void copyToHost(GPUTaskState *dst, GPUTaskState *src, size_t NUM_TASKS)
{
    float *hostData = (float*)src;

    for (int i = 0; i < NUM_TASKS; i++)
    {
        GPUTaskState curr;
        for (int j = 0; j < sizeof(GPUTaskState) / sizeof(float); j++)
        {
            ((float*)&curr)[j] = hostData[j * NUM_TASKS + i];
        }
        dst[i] = curr;
    }
}

// Init state buffers (rays, tasks) needed by microkernels
void CLContext::initMCBuffers()
{
    // TODO: ensure 32bit divisibility in SoA mode
    const size_t t_bytes = NUM_TASKS * sizeof(GPUTaskState);
    deviceBuffers.tasksBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, t_bytes, NULL, &err);
    verify("Task buffer creation failed!");

    // Queues
    cl_uint pixelIndex = 0;

    // TODO: CL_MEM_USE_HOST_PTR for seeing queues on host
    deviceBuffers.currentPixelIdx = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1 * sizeof(cl_uint), (void*)&pixelIndex, &err);
    deviceBuffers.queueCounters = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(QueueCounters), (void*)&hostCounters, &err);
    deviceBuffers.raygenQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.extensionQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.shadowQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.diffuseMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.glossyMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.ggxReflMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.ggxRefrMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    deviceBuffers.deltaMatQueue = cl::Buffer(context, CL_MEM_READ_WRITE, NUM_TASKS * sizeof(cl_uint), NULL, &err);
    verify("MK queue creation failed");

    const size_t memoryUsageMiB = t_bytes / (2 << 19);
    std::cout << "Microkernel state data: " << memoryUsageMiB << " MiB" << std::endl;
}

void CLContext::setKernelBuildSettings()
{
    std::string buildOpts = "-DGPU -I./src -cl-denorms-are-zero -cl-fast-relaxed-math -cl-kernel-arg-info -DFLT_FLOAT_ATOMICS";
    Settings &s = Settings::getInstance();
    if (s.getUseBitstack()) buildOpts += " -DUSE_BITSTACK";
    if (s.getUseSoA()) buildOpts += " -DUSE_SOA";
    if (platformIsNvidia(platform)) buildOpts += " -cl-nv-verbose";

    // Static, shared by all kernels
    flt::Kernel::setBuildOptions(buildOpts);
}

void CLContext::setupPickKernel()
{
    if (!kernel_pick)
        kernel_pick = new PickKernel();

    window->showMessage("Building kernel", "kernel_pick");
    kernel_pick->build("src/kernel_pick.cl", "pick", context, device, platform);
}

void CLContext::setupWfExtKernel()
{
    if (!wf_extension)
        wf_extension = new WFExtensionKernel();

    window->showMessage("Building kernel", "wf_extrays");
    wf_extension->build("src/wf_extrays.cl", "traceExtension", context, device, platform);
}

void CLContext::setupWfLogicKernel()
{
    if (!wf_logic)
        wf_logic = new WFLogicKernel();

    window->showMessage("Building kernel", "wf_logic");
    wf_logic->build("src/wf_logic.cl", "logic", context, device, platform);
}

void CLContext::setupWfShadowKernel()
{
    if (!wf_shadow)
        wf_shadow = new WFShadowKernel();

    window->showMessage("Building kernel", "wf_shadowrays");
    wf_shadow->build("src/wf_shadowrays.cl", "traceShadow", context, device, platform);
}

void CLContext::setupWfRaygenKernel()
{
    if (!wf_raygen)
        wf_raygen = new WFRaygenKernel();

    window->showMessage("Building kernel", "wf_raygen");
    wf_raygen->build("src/wf_raygen.cl", "genRays", context, device, platform);
}

void CLContext::setupWfDiffuseKernel()
{
    if (!wf_diffuse)
        wf_diffuse = new WFDiffuseKernel();

    window->showMessage("Building kernel", "wf_mat_diffuse");
    wf_diffuse->build("src/wf_mat_diffuse.cl", "wavefrontDiffuse", context, device, platform);
}

void CLContext::setupWfGlossyKernel()
{
    if (!wf_glossy)
        wf_glossy = new WFGlossyKernel();

    window->showMessage("Building kernel", "wf_mat_glossy");
    wf_glossy->build("src/wf_mat_glossy.cl", "wavefrontGlossy", context, device, platform);
}

void CLContext::setupWfGGXReflKernel()
{
    if (!wf_ggx_refl)
        wf_ggx_refl = new WFGGXReflKernel();

    window->showMessage("Building kernel", "wf_mat_ggx_reflection");
    wf_ggx_refl->build("src/wf_mat_ggx_reflection.cl", "wavefrontGGXReflection", context, device, platform);
}

void CLContext::setupWfGGXRefrKernel()
{
    if (!wf_ggx_refr)
        wf_ggx_refr = new WFGGXRefrKernel();

    window->showMessage("Building kernel", "wf_mat_ggx_refraction");
    wf_ggx_refr->build("src/wf_mat_ggx_refraction.cl", "wavefrontGGXRefraction", context, device, platform);
}

void CLContext::setupWfDeltaKernel()
{
    if (!wf_delta)
        wf_delta = new WFDeltaKernel();

    window->showMessage("Building kernel", "wf_mat_delta");
    wf_delta->build("src/wf_mat_delta.cl", "wavefrontDelta", context, device, platform);
}

void CLContext::setupWfAllMaterialsKernel()
{
    if (!wf_mat_all)
        wf_mat_all = new WFAllMaterialsKernel();

    window->showMessage("Building kernel", "wf_mat_all");
    wf_mat_all->build("src/wf_mat_all.cl", "wavefrontAllMaterials", context, device, platform);
}

void CLContext::setupWfResetKernel()
{
    if (!wf_reset)
        wf_reset = new WFResetKernel();
    
    window->showMessage("Building kernel", "wf_reset");
    wf_reset->build("src/wf_reset.cl", "reset", context, device, platform);
}

void CLContext::setupResetKernel()
{
    if (!mk_reset)
        mk_reset = new MKResetKernel();

    window->showMessage("Building kernel", "mk_reset");
    mk_reset->build("src/mk_reset.cl", "reset", context, device, platform);
}

void CLContext::setupRayGenKernel()
{
    if (!mk_raygen)
        mk_raygen = new MKRaygenKernel();

    window->showMessage("Building kernel", "mk_raygen");
    mk_raygen->build("src/mk_raygen.cl", "genCameraRays", context, device, platform);
}

void CLContext::setupNextVertexKernel()
{
    if (!mk_next_vertex)
        mk_next_vertex = new MKNextVertexKernel();

    window->showMessage("Building kernel", "mk_next_vertex");
    mk_next_vertex->build("src/mk_next_vertex.cl", "nextVertex", context, device, platform);
}

void CLContext::setupBsdfSampleKernel()
{
    if (!mk_sample_bsdf)
        mk_sample_bsdf = new MKSampleBSDFKernel();

    window->showMessage("Building kernel", "mk_sample_bsdf");
    mk_sample_bsdf->build("src/mk_sample_bsdf.cl", "sampleBsdf", context, device, platform);
}

void CLContext::setupSplatKernel()
{
    if (!mk_splat)
        mk_splat = new MKSplatKernel();

    window->showMessage("Building kernel", "mk_splat");
    mk_splat->build("src/mk_splat.cl", "splat", context, device, platform);
}

void CLContext::setupSplatPreviewKernel()
{
    if (!mk_splat_preview)
        mk_splat_preview = new MKSplatPreviewKernel();

    window->showMessage("Building kernel", "mk_splat_preview");
    mk_splat_preview->build("src/mk_splat_preview.cl", "splatPreview", context, device, platform);
}

void CLContext::setupPostprocessKernel()
{
    if (!mk_postprocess)
        mk_postprocess = new MKPostprocessKernel();

    window->showMessage("Building kernel", "mk_postprocess");
    mk_postprocess->build("src/mk_postprocess.cl", "process", context, device, platform);
}

void CLContext::setupPixelStorage(PTWindow *window)
{
    if (sharedMemory.size() > 0)
    {
        sharedMemory.clear(); // memory freed by cl-cpp-wrapper
    }

    GLuint *tex_arr = window->getTexPtr();
    unsigned int numPixels = window->getTexWidth() * window->getTexHeight();

    deviceBuffers.pixelBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(cl_float) * 4, NULL, &err); // microkernel pixel buffer
    deviceBuffers.denoiserAlbedoBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(cl_float) * 4, NULL, &err);
    deviceBuffers.denoiserNormalBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(cl_float) * 4, NULL, &err);
    deviceBuffers.previewBuffer = cl::BufferGL(context, CL_MEM_READ_WRITE, window->getPBO(), &err); // GL preview buffer
    deviceBuffers.denoiserAlbedoBufferGL = cl::BufferGL(context, CL_MEM_READ_WRITE, window->getAlbedoPBO(), &err);
    deviceBuffers.denoiserNormalBufferGL = cl::BufferGL(context, CL_MEM_READ_WRITE, window->getNormalPBO(), &err);
    sharedMemory = { deviceBuffers.previewBuffer, deviceBuffers.denoiserAlbedoBufferGL, deviceBuffers.denoiserNormalBufferGL };
    verify("CL pixel storage creation failed!");

	// Set new kernel args (pointers might have changed)
	err = 0;
	if (mk_splat)
		err |= mk_splat->setArg("pixels", deviceBuffers.pixelBuffer);
    if (mk_splat_preview)
        err |= mk_splat_preview->setArg("pixels", deviceBuffers.pixelBuffer);
    if (mk_next_vertex)
        err |= mk_next_vertex->setArg("denoiserNormal", deviceBuffers.denoiserNormalBuffer);
    if (mk_sample_bsdf)
        err |= mk_sample_bsdf->setArg("denoiserAlbedo", deviceBuffers.denoiserAlbedoBuffer);
    if (mk_reset)
    {
        err |= mk_reset->setArg("pixels", deviceBuffers.pixelBuffer);
        err |= mk_reset->setArg("denoiserAlbedo", deviceBuffers.denoiserAlbedoBuffer);
        err |= mk_reset->setArg("denoiserNormal", deviceBuffers.denoiserNormalBuffer);
    }
    if (wf_logic)
    {
        err |= wf_logic->setArg("pixels", deviceBuffers.pixelBuffer);
        err |= wf_logic->setArg("denoiserNormal", deviceBuffers.denoiserNormalBuffer);
        err |= wf_logic->setArg("denoiserAlbedo", deviceBuffers.denoiserAlbedoBuffer);
    }
    if (wf_reset)
    {
        err |= wf_reset->setArg("pixels", deviceBuffers.pixelBuffer);
        err |= wf_reset->setArg("denoiserAlbedo", deviceBuffers.denoiserAlbedoBuffer);
        err |= wf_reset->setArg("denoiserNormal", deviceBuffers.denoiserNormalBuffer);
    }
    if (mk_postprocess)
    {
        err |= mk_postprocess->setArg("pixelsRaw", deviceBuffers.pixelBuffer);
        err |= mk_postprocess->setArg("denoiserAlbedo", deviceBuffers.denoiserAlbedoBuffer);
        err |= mk_postprocess->setArg("denoiserNormal", deviceBuffers.denoiserNormalBuffer);
        err |= mk_postprocess->setArg("pixelsPreview", deviceBuffers.previewBuffer);
        err |= mk_postprocess->setArg("denoiserAlbedoGL", deviceBuffers.denoiserAlbedoBufferGL);
        err |= mk_postprocess->setArg("denoiserNormalGL", deviceBuffers.denoiserNormalBufferGL);
    }
        
	verify("Failed to update kernel pixel storage args");
}

void CLContext::saveImage(std::string filename, const RenderParams &params)
{
    unsigned int numBytes = params.width * params.height * 3; // rgb
    unsigned int numFloats = params.width * params.height * 4; // rgba
	std::unique_ptr<unsigned char[]> dataBytes(new unsigned char[numBytes]);
    std::unique_ptr<float[]> dataFloats(new float[numFloats]);

    bool hdr = endsWith(filename, ".hdr") || endsWith(filename, ".HDR");

    glFinish();

    // Copy data to host
    err = 0;
    err |= cmdQueue.enqueueAcquireGLObjects(&sharedMemory);

    cl::Buffer &pixels = (hdr) ? deviceBuffers.pixelBuffer : deviceBuffers.previewBuffer;
    err |= cmdQueue.enqueueReadBuffer(pixels, CL_TRUE, 0, numFloats * sizeof(float), dataFloats.get());
    err |= cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    err |= cmdQueue.finish();
    verify("Failed to copy pixel buffer to host!");
    
    if (hdr)
    {
        // Save linear unclamped values
        for (int i = 0; i < numFloats; i += 4)
        {
            
            float *r = &dataFloats[i] + 0;
            float *g = &dataFloats[i] + 1;
            float *b = &dataFloats[i] + 2;
            float *a = &dataFloats[i] + 3;

            *r /= *a;
            *g /= *a;
            *b /= *a;
            *a = 1.0f;
        }

        ILuint imageID = ilGenImage();
        ilBindImage(imageID);
        ilTexImage(params.width, params.height, 1, 4, IL_RGBA, IL_FLOAT, dataFloats.get());
        ilSaveImage(filename.c_str());
        ilDeleteImage(imageID);
    }
    else
    {
        // Convert to bytes
        // Already tonemapped and gamma-corrected
        int counter = 0;
        for (int i = 0; i < numFloats; i += 4)
        {
            float r = dataFloats[i + 0];
            float g = dataFloats[i + 1];
            float b = dataFloats[i + 2];
            float a = dataFloats[i + 3];

            // Convert to bytes
            auto clamp = [](float value) { return std::max(0.0f, std::min(1.0f, value)); };
            dataBytes[counter++] = (unsigned char)(255 * clamp(r));
            dataBytes[counter++] = (unsigned char)(255 * clamp(g));
            dataBytes[counter++] = (unsigned char)(255 * clamp(b));
        }

        ILuint imageID = ilGenImage();
        ilBindImage(imageID);
        ilTexImage(params.width, params.height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, dataBytes.get());
        ilSaveImage(filename.c_str());
        ilDeleteImage(imageID);
    }
    

    // Check for errors
    ILenum Error = IL_NO_ERROR;
    while ((Error = ilGetError()) != IL_NO_ERROR)
    {
        printf("\n%d: %s", Error, iluErrorString(Error));
    }
	
	std::cout << ((Error == IL_NO_ERROR) ? "\nSaved " : "\nFailed saving ") << filename << std::endl;
}

void CLContext::createEnvMap(EnvironmentMap *map)
{
	int width = map->getWidth(), height = map->getHeight();
	float *data = map->getData();

    // Convert rgb to rgba (OpenCL doesn't support floats for RGB-images)
    float *rgba = new float[width * height * 4];
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // RGB
            for (int c = 0; c < 3; c++) // color channels
            {
                rgba[(h * width + w) * 4 + c] = data[(h * width + w) * 3 + c];
            }
            // Alpha
            rgba[(h * width + w) * 4 + 3] = 1.0f;
        }
    }

	// Upload rgb colors
    const cl::ImageFormat format(CL_RGBA, CL_FLOAT);
    deviceBuffers.environmentMap = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width, height, 0, rgba, &err);
    verify("Environment map creation failed!");

	// Upload probability and alias tables for importance sampling
	size_t pBytes = width * height * sizeof(float);
	size_t aBytes = width * height * sizeof(int);
    deviceBuffers.probTable = cl::Buffer(context, CL_MEM_READ_ONLY, pBytes, NULL, &err);
    deviceBuffers.aliasTable = cl::Buffer(context, CL_MEM_READ_ONLY, aBytes, NULL, &err);
    deviceBuffers.pdfTable = cl::Buffer(context, CL_MEM_READ_ONLY, pBytes, NULL, &err);
	verify("Env map IS table creation failed");

	err |= cmdQueue.enqueueWriteBuffer(deviceBuffers.probTable, CL_TRUE, 0, pBytes, map->getProbTable());
	err |= cmdQueue.enqueueWriteBuffer(deviceBuffers.aliasTable, CL_TRUE, 0, aBytes, map->getAliasTable());
	err |= cmdQueue.enqueueWriteBuffer(deviceBuffers.pdfTable, CL_TRUE, 0, pBytes, map->getPdfTable());
	verify("Env map IS table writing failed");

	// Cleanup
    delete[] rgba;

    // Update env map references
    setupKernels();
}

void CLContext::setupScene()
{
	// Dummy env map
	float rgba[4] { 0.0f, 0.0f, 0.0f, 0.0f };
    deviceBuffers.environmentMap = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_FLOAT), 1, 1, 0, rgba, &err);
	verify("Dummy env map creation failed");
}

// Upload BVH data, geometry and materials to GPU
void CLContext::uploadSceneData(BVH *bvh, Scene *scene)
{
    std::vector<RTTriangle> *tris = bvh->m_triangles;
    std::vector<cl_uint> *indices = &bvh->m_indices; 
    std::vector<Node> *nodes = &bvh->m_nodes;
    std::vector<Material> *materials = &scene->getMaterials();

    size_t t_bytes = tris->size() * sizeof(RTTriangle);
    size_t i_bytes = indices->size() * sizeof(cl_uint);
    size_t n_bytes = nodes->size() * sizeof(Node);
    size_t m_bytes = materials->size() * sizeof(Material);

    // Allocate memory for buffers
    deviceBuffers.triangleBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, t_bytes, NULL, &err);
    verify("Triangle buffer creation failed!");

    deviceBuffers.indexBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, i_bytes, NULL, &err);
    verify("Index buffer creation failed!");

    deviceBuffers.nodeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, n_bytes, NULL, &err);
    verify("Node buffer creation failed!");

    if(m_bytes > 0) deviceBuffers.materialBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, m_bytes, NULL, &err);
    verify("Material buffer creation failed!");


    // Write data to buffers
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.triangleBuffer, CL_TRUE, 0, t_bytes, tris->data());
    verify("Triangle buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.indexBuffer, CL_TRUE, 0, i_bytes, indices->data());
    verify("Index buffer writing failed!");

    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.nodeBuffer, CL_TRUE, 0, n_bytes, nodes->data());
    verify("Node buffer writing failed!");

    if(m_bytes > 0) err = cmdQueue.enqueueWriteBuffer(deviceBuffers.materialBuffer, CL_TRUE, 0, m_bytes, materials->data());
    verify("Material buffer writing failed!");

    // Pack texture data into aggregate array
    packTextures(scene);

    // Ensures that the kernels have the correct arguments
    setupKernels();
}

// Upload texture data to GPU
// Avoids intermediate buffers to keep RAM usage low
void CLContext::packTextures(Scene *scene)
{
    std::vector<Texture*> textures = scene->getTextures();

    if (textures.size() == 0) return;
    
    // Calculate total size required for texture data
    size_t t_bytes = 0;
    for (Texture *tex : textures)
    {
        t_bytes += tex->getWidth() * tex->getHeight() * 4 * 1; // RGBA
    }

    // Create buffers for texture data & descriptors
    size_t d_bytes = textures.size() * sizeof(TexDescriptor);
    deviceBuffers.texDescriptorBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, d_bytes, NULL, &err);
    verify("Texture descriptor buffer creation failed!");
    deviceBuffers.texDataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, t_bytes, NULL, &err);
    verify("Texture data buffer creation failed!");
    
    // Upload data, create descriptors
    std::vector<TexDescriptor> descs;
    cl_uint offset = 0;
    for (Texture *tex : textures)
    {
        TexDescriptor desc;
        desc.offset = offset;
        desc.width = tex->getWidth();
        desc.height = tex->getHeight();
        descs.push_back(desc);

        cl_uint len = tex->getWidth() * tex->getHeight() * 4 * 1; // RGBA
        err = cmdQueue.enqueueWriteBuffer(deviceBuffers.texDataBuffer, CL_TRUE, offset, len, tex->getData());
        verify("Texture data buffer writing failed!");

        offset += len;
    }

    // Upload descriptors
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.texDescriptorBuffer, CL_TRUE, 0, d_bytes, descs.data());
    verify("Texture descriptor buffer writing failed!");
}

// Passing structs to kernels is broken in several drivers (e.g. GT 750M on MacOS)
// Allocating memory for the rendering params is more compatible
void CLContext::setupParams()
{
    deviceBuffers.renderParams = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(RenderParams) * 1, NULL, &err);
    verify("Params buffer creation failed!");
}

void CLContext::setupPickResult()
{
    deviceBuffers.pickResult = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(Hit) * 1, NULL, &err);
    verify("Pick result creation failed!");
}

void CLContext::setupStats()
{
    deviceBuffers.renderStats = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(RenderStats) * 1, NULL, &err);
    verify("RenderStats creation failed!");
    resetStats();
}

void CLContext::resetStats()
{
    RenderStats s = { 0, 0, 0, 0 };
    statsAsync = s;
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.renderStats, CL_TRUE, 0, sizeof(RenderStats), &s);
    verify("Stats buffer reset failed!");
}

void CLContext::fetchStatsAsync()
{
    err = cmdQueue.enqueueReadBuffer(deviceBuffers.renderStats, CL_FALSE, 0, sizeof(RenderStats), &statsAsync);
    verify("Failed to enqueue async stat transfer!");
}

void CLContext::updateRenderPerf(float deltaT)
{
    double scale = 1e6 * deltaT;
    renderPerf.primary = statsAsync.primaryRays / scale;
    renderPerf.extension = statsAsync.extensionRays / scale;
    renderPerf.shadow = statsAsync.shadowRays / scale;
    renderPerf.samples = statsAsync.samples / scale;
    renderPerf.total = renderPerf.primary + renderPerf.extension + renderPerf.shadow;
}

const PerfNumbers CLContext::getRenderPerf()
{
    return renderPerf;
}

const RenderStats CLContext::getStats()
{
    return statsAsync;
}

void CLContext::enqueueGetCounters(QueueCounters *cnt)
{
    err = cmdQueue.enqueueReadBuffer(deviceBuffers.queueCounters, CL_FALSE, 0, 1 * sizeof(QueueCounters), cnt);
}

void CLContext::checkTracingPerf()
{
    // Check ray tracing perf without overhead
    cl_ulong t0Ext, t0Shadow;
    cl_ulong t1Ext, t1Shadow;

    clGetEventProfilingInfo(extRayEvent(), CL_PROFILING_COMMAND_START, sizeof(t0Ext), &t0Ext, NULL);
    clGetEventProfilingInfo(extRayEvent(), CL_PROFILING_COMMAND_END, sizeof(t1Ext), &t1Ext, NULL);
    clGetEventProfilingInfo(shdwRayEvent(), CL_PROFILING_COMMAND_START, sizeof(t0Shadow), &t0Shadow, NULL);
    clGetEventProfilingInfo(shdwRayEvent(), CL_PROFILING_COMMAND_END, sizeof(t1Shadow), &t1Shadow, NULL);

    // Extension rays
    double timeNs = t1Ext - t0Ext;
    double timeMs = timeNs / 1000000.0;
    double timeS = timeMs / 1000.0;

    double scale = 1e6 * timeS;
    double MRaysExt = statsAsync.extensionRays / scale;
    printf("Ext ray time: %0.3f milliseconds, speed: %.2f MRays/s \n", timeMs, MRaysExt);

    // Shadow rays
    timeNs = t1Shadow - t0Shadow;
    timeMs = timeNs / 1000000.0;
    timeS = timeMs / 1000.0;

    scale = 1e6 * timeS;
    double MRaysShadow = statsAsync.shadowRays / scale;
    printf("Shadow ray time: %0.3f milliseconds, speed: %.2f MRays/s \n", timeMs, MRaysShadow);
}

void CLContext::updateParams(const RenderParams &params)
{
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.renderParams, CL_FALSE, 0, sizeof(RenderParams), &params);
    verify("RenderParam writing failed");
}

void CLContext::enqueueResetKernel(const RenderParams &params)
{
	err = 0;
	err |= cmdQueue.enqueueNDRangeKernel(mk_reset->getKernel(), cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
	verify("Failed to enqueue reset kernel!");
}

void CLContext::enqueueRayGenKernel(const RenderParams &params)
{
    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_raygen->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue ray gen kernel!");
}

void CLContext::enqueueNextVertexKernel(const RenderParams &params)
{
    // Enqueue 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_next_vertex->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue next vertex kernel!");
}

void CLContext::enqueueBsdfSampleKernel(const RenderParams &params)
{
    // Enqueue 1D range
    err = 0;
    err = cmdQueue.enqueueNDRangeKernel(mk_sample_bsdf->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue bsdf sample kernel!");
}

void CLContext::enqueueSplatKernel(const RenderParams &params)
{
    // TODO: find out why my GTX 780 won't enqueue 1D kernels! (due to image2d_type?)
    // TODO: also, look at having global wg be a multiple of local wg (or a multiple of 32/64)
    err = cmdQueue.enqueueNDRangeKernel(mk_splat->getKernel(), cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat kernel!");
}

void CLContext::enqueueSplatPreviewKernel(const RenderParams &params)
{
    err = cmdQueue.enqueueNDRangeKernel(mk_splat_preview->getKernel(), cl::NullRange, cl::NDRange(params.width, params.height), cl::NullRange);
    verify("Failed to enqueue splat preview kernel!");
}

void CLContext::enqueuePostprocessKernel(const RenderParams & params)
{   
    err = cmdQueue.enqueueAcquireGLObjects(&sharedMemory);
    verify("Failed to enqueue GL object acquisition!");

    // 1D range
    err = cmdQueue.enqueueNDRangeKernel(mk_postprocess->getKernel(), cl::NullRange, cl::NDRange(params.width * params.height), cl::NullRange);
    verify("Failed to enqueue postprocess kernel!");

    err = cmdQueue.enqueueReleaseGLObjects(&sharedMemory);
    verify("Failed to enqueue GL object release!");
}

void CLContext::enqueueWfResetKernel(const RenderParams & params)
{
    cl_uint numElems = std::max(NUM_TASKS, params.width * params.height);
    err = cmdQueue.enqueueNDRangeKernel(wf_reset->getKernel(), cl::NullRange, cl::NDRange(numElems), cl::NullRange);
    verify("Failed to enqueue wf_reset");
}

void CLContext::enqueueWfRaygenKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_raygen->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_raygen");
}

void CLContext::enqueueWfExtRayKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_extension->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange, 0, &extRayEvent);
    verify("Failed to enqueue wf_extension");
}

void CLContext::enqueueWfShadowRayKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_shadow->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange, 0, &shdwRayEvent);
    verify("Failed to enqueue wf_shadow");
}

void CLContext::enqueueWfLogicKernel(const RenderParams& params, const bool firstIteration)
{
    cl_uint numElems = (firstIteration) ? std::max(NUM_TASKS, params.width * params.height) : NUM_TASKS;
    err |= wf_logic->setArg("firstIteration", (cl_uint)firstIteration);
    err |= cmdQueue.enqueueNDRangeKernel(wf_logic->getKernel(), cl::NullRange, cl::NDRange(numElems), cl::NullRange);
    verify("Failed to enqueue wf_logic");
}

void CLContext::enqueueWfMaterialKernels(const RenderParams & params)
{
    if (params.wfSeparateQueues)
    {
        enqueueWfDiffuseKernel(params);
        enqueueWfGlossyKernel(params);
        enqueueWfGGXReflKernel(params);
        enqueueWfGGXRefrKernel(params);
        enqueueWfDeltaKernel(params);
    }
    else
    {
        enqueueWfAllMaterialsKernel(params);
    }
}

void CLContext::enqueueWfDiffuseKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_diffuse->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_diffuse");
}

void CLContext::enqueueWfGlossyKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_glossy->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_glossy");
}

void CLContext::enqueueWfGGXReflKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_ggx_refl->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_ggx_refl");
}

void CLContext::enqueueWfGGXRefrKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_ggx_refr->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_ggx_refr");
}

void CLContext::enqueueWfDeltaKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_delta->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_delta");
}

void CLContext::enqueueWfAllMaterialsKernel(const RenderParams & params)
{
    err = cmdQueue.enqueueNDRangeKernel(wf_mat_all->getKernel(), cl::NullRange, cl::NDRange(NUM_TASKS), cl::NullRange);
    verify("Failed to enqueue wf_mat_all");
}

// Only recompiles kernels that need recompiling
// Param setArgs defines if kernel arguments are set even if kernel isn't recompiled
void CLContext::recompileKernels(bool setArgs)
{
    kernel_pick->rebuild(setArgs);
    mk_postprocess->rebuild(setArgs);
    
    wf_reset->rebuild(setArgs);
    wf_extension->rebuild(setArgs);
    wf_raygen->rebuild(setArgs);
    wf_logic->rebuild(setArgs);
    wf_shadow->rebuild(setArgs);
    wf_diffuse->rebuild(setArgs);
    wf_glossy->rebuild(setArgs);
    wf_ggx_refl->rebuild(setArgs);
    wf_ggx_refr->rebuild(setArgs);
    wf_delta->rebuild(setArgs);

    mk_reset->rebuild(setArgs);
    mk_raygen->rebuild(setArgs);
    mk_next_vertex->rebuild(setArgs);
    mk_sample_bsdf->rebuild(setArgs);
    mk_splat->rebuild(setArgs);
    mk_splat_preview->rebuild(setArgs);
}

// Clear wavefront queues by setting counters to zero
void CLContext::enqueueClearWfQueues()
{
    QueueCounters empty = {};
    hostCounters = empty;
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.queueCounters, CL_FALSE, 0, sizeof(QueueCounters), &hostCounters);
    verify("Failed to enqueue wavefront queueCounter read");
}

void CLContext::finishQueue()
{
    err = cmdQueue.finish();
    verify("Failed to finish command queue!");
}

void CLContext::updatePixelIndex(cl_uint numPixels, cl_uint numNewPaths)
{
    pixelIdx = (pixelIdx + numNewPaths) % numPixels;
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.currentPixelIdx, CL_FALSE, 0, sizeof(cl_uint), &pixelIdx); // will be available when raygen runs
}

void CLContext::resetPixelIndex()
{
    pixelIdx = 0;
    err = cmdQueue.enqueueWriteBuffer(deviceBuffers.currentPixelIdx, CL_FALSE, 0, sizeof(cl_uint), &pixelIdx);
}

cl_uint CLContext::getNumTasks() const
{
    return NUM_TASKS;
}

Hit CLContext::pickSingle(float NDCx, float NDCy)
{
    err = 0;
    err |= kernel_pick->setArg("NDCx", NDCx);
    err |= kernel_pick->setArg("NDCy", NDCy);
    verify("Failed to set pick kernel coordinates");

    Hit hit;

    err |= cmdQueue.enqueueNDRangeKernel(kernel_pick->getKernel(), cl::NullRange, cl::NDRange(1), cl::NullRange);
    err |= cmdQueue.enqueueReadBuffer(deviceBuffers.pickResult, CL_FALSE, 0, 1 * sizeof(Hit), &hit);
    cmdQueue.finish();
    verify("Failed to execute pick kernel or get result");

    return hit;
}

cl::Platform &CLContext::getPlatformByName(std::vector<cl::Platform> &platforms, std::string name) {
    for (cl::Platform &p : platforms) {
        std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
        if (platformName.find(name) != std::string::npos) {
            return p;
        }
    }

    std::cout << "No platform name containing \"" << name << "\" found!" << std::endl;
    return platforms[0];
}

cl::Device &CLContext::getDeviceByName(std::vector<cl::Device> &devices, std::string name) {
    for (cl::Device &d : devices) {
        std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
        if (deviceName.find(name) != std::string::npos) {
            return d;
        }
    }

    std::cout << "No device name containing \"" << name << "\" in selected context!" << std::endl;
    return devices[0];
}

// Check error, second optional parameter acts as boolean predicate
void CLContext::verify(std::string msg, int pred)
{
	// Use default value if predicate is -1
	int code = (pred > -1) ? !pred : this->err;

    if(code != CL_SUCCESS)
    {
        std::string message = msg + " (" + getCLErrorString(code) + ")";
        std::cout << message << std::endl;
        waitExit();
    }
}

// Print the devices, C++ style
void CLContext::printDevices()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    const std::string DECORATOR = "================";

    int platform_id = 0;
    int device_id = 0;

    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    for(cl::Platform &platform : platforms)
    {
        std::cout << DECORATOR << " Platform " << platform_id++ << " (" << platform.getInfo<CL_PLATFORM_NAME>() << ") " << DECORATOR << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        for(cl::Device &device : devices)
        {
            bool GPU = (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);

            std::cout << "Device " << device_id++ << ": " << std::endl;
            std::cout << "\tName: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "\tType: " << (GPU ? "(GPU)" : "(CPU)") << std::endl;
            std::cout << "\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\tCompute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\tGlobal Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
            std::cout << "\tMax Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            std::cout << "\tMax Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            std::cout << "\tLocal Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            std::cout << "\tAvailable: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
        }
        std::cout << std::endl;
    }
}



