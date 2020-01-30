#pragma once

//#include <optixu/optixpp_namespace.h>


#include "CUDABuffer.h"

using osc::CUDABuffer;

class PTWindow;
class DenoiserOptix
{
public:
    DenoiserOptix(void);
    ~DenoiserOptix(void) = default;

    void bindBuffers(PTWindow *window);
    void resizeBuffers(PTWindow *window);
    void denoise(void);

    void setBlend(float val);

private:
    void setupDenoiser(unsigned int width, unsigned int height);

    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    OptixDeviceContext optixContext;

    cudaGraphicsResource_t handleColor;
    cudaGraphicsResource_t handleNormal;
    cudaGraphicsResource_t handleAlbedo;

    OptixDenoiser denoiser = nullptr; // optix7 internal type
    CUDABuffer    denoiserScratch;
    CUDABuffer    denoiserState;
    CUDABuffer    denoiserIntensity;

    unsigned int fbWidth;
    unsigned int fbHeight;

    // Amount of original image blended into denoised result, in range [0.0, 1.0]
    float denoiseBlend = 0.0f;
};
