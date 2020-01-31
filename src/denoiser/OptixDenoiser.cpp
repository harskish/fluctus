#define NOMINMAX

#include "../window.hpp"
#include "../utils.h"
#include <algorithm>
#include <iostream>
#include <cuda_gl_interop.h>
#include "OptixDenoiser.hpp"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

DenoiserOptix::DenoiserOptix(void)
{
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("OptixDenoiser: no CUDA capable devices found!");
    std::cout << "OptixDenoiser: found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit());
    std::cout << "OptixDenoiser: successfully initialized optix... yay!" << std::endl;

    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "OptixDenoiser: running on device: " << deviceProps.name << std::endl;

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

// Create RTBuffers using CUDA-GL sharing
// The buffers are now doubly shared (CUDA-GL and CL-GL)
void DenoiserOptix::bindBuffers(PTWindow* window)
{
    unsigned int width = window->getTexWidth();
    unsigned int height = window->getTexHeight();

    auto create = [&](GLuint pbo, cudaGraphicsResource_t &cudaResourceBuf, cudaGraphicsRegisterFlags flags)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);

        if (cudaSuccess != cudaGraphicsGLRegisterBuffer(&cudaResourceBuf, pbo, flags)) // matches CL_MEM_READ_WRITE?
            throw std::runtime_error("Could not create GL-CUDA shared PBO");

        // map OpenGL buffer object for writing from CUDA
        float4* dptr;

        if (cudaSuccess != cudaGraphicsMapResources(1, &cudaResourceBuf, 0))
            throw std::runtime_error("Could not map resource");

        size_t expected_bytes = width * height * 4 * sizeof(float);
        size_t num_bytes;
        if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cudaResourceBuf))
            throw std::runtime_error("Could not get mapped ptr");

        printf("CUDA mapped VBO: May access %ld bytes (expected: %ld)\n", num_bytes, expected_bytes);

        // unmap buffer object
        if (cudaSuccess != cudaGraphicsUnmapResources(1, &cudaResourceBuf, 0))
            throw std::runtime_error("Could not unmap resource");
    };

    create(window->getPBO(), handleColor, cudaGraphicsRegisterFlagsNone); // read and write
    create(window->getNormalPBO(), handleNormal, cudaGraphicsRegisterFlagsReadOnly);
    create(window->getAlbedoPBO(), handleAlbedo, cudaGraphicsRegisterFlagsReadOnly);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    setupDenoiser(width, height);
}

// Called on framebuffer resize
void DenoiserOptix::resizeBuffers(PTWindow* window)
{
    fbWidth = window->getTexWidth();
    fbHeight = window->getTexHeight();
    bindBuffers(window);
}

// Perform denoising, write results to GL buffer
void DenoiserOptix::denoise(void)
{
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 1;
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    denoiserParams.blendFactor = 1.0f - denoiseBlend;

    auto mapBuffer = [&](cudaGraphicsResource_t& cudaResourceBuf)
    {
        float4* dptr;

        if (cudaSuccess != cudaGraphicsMapResources(1, &cudaResourceBuf, 0))
            throw std::runtime_error("Could not map resource");

        size_t num_bytes;
        if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cudaResourceBuf))
            throw std::runtime_error("Could not get mapped ptr");

        return (CUdeviceptr)dptr;
    };

    auto unmapBuffer = [&](cudaGraphicsResource_t& cudaResourceBuf)
    {
        if (cudaSuccess != cudaGraphicsUnmapResources(1, &cudaResourceBuf, 0))
            throw std::runtime_error("Could not unmap resource");
    };

    // MAP GL BUFFERS
    CUdeviceptr color = mapBuffer(handleColor);
    CUdeviceptr albedo = mapBuffer(handleAlbedo);
    CUdeviceptr normal = mapBuffer(handleNormal);

    OptixImage2D inputLayer[3];
    CUdeviceptr ptrs[3] { color, albedo, normal };
    for (int i = 0; i < 3; i++)
    {
        inputLayer[i].data = ptrs[i];
        inputLayer[i].width = fbWidth;
        inputLayer[i].height = fbHeight;
        inputLayer[i].rowStrideInBytes = fbWidth * sizeof(float4);
        inputLayer[i].pixelStrideInBytes = sizeof(float4);
        inputLayer[i].format = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    OptixImage2D outputLayer;
    outputLayer.data = color; // Overwrite!
    outputLayer.width = fbWidth;
    outputLayer.height = fbHeight;
    outputLayer.rowStrideInBytes = fbWidth * sizeof(float4);
    outputLayer.pixelStrideInBytes = sizeof(float4);
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser,
        /*stream*/0,
        &inputLayer[0],
        (CUdeviceptr)denoiserIntensity.d_pointer(),
        (CUdeviceptr)denoiserScratch.d_pointer(),
        denoiserScratch.size()));

    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
        /*stream*/0,
        &denoiserParams,
        denoiserState.d_pointer(),
        denoiserState.size(),
        &inputLayer[0], 2,
        /*inputOffsetX*/0,
        /*inputOffsetY*/0,
        &outputLayer,
        denoiserScratch.d_pointer(),
        denoiserScratch.size()));

    // UNMAP GL BUFFERS
    unmapBuffer(handleColor);
    unmapBuffer(handleAlbedo);
    unmapBuffer(handleNormal);

    CUDA_SYNC_CHECK();
}

void DenoiserOptix::setBlend(float val)
{
    denoiseBlend = std::max(0.0f, std::min(val, 1.0f));
}

void DenoiserOptix::setupDenoiser(unsigned int width, unsigned int height)
{
    if (denoiser)
        OPTIX_CHECK(optixDenoiserDestroy(denoiser));

    OptixDenoiserOptions denoiserOptions;
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO; //OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, NULL, 0));

    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height,
        &denoiserReturnSizes));

    denoiserIntensity.resize(sizeof(float));
    denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

    this->fbWidth = width;
    this->fbHeight = height;

    OPTIX_CHECK(optixDenoiserSetup(denoiser, 0,
        fbWidth, fbHeight,
        denoiserState.d_pointer(),
        denoiserState.size(),
        denoiserScratch.d_pointer(),
        denoiserScratch.size()));
}
