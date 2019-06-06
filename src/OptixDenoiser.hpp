#pragma once

#include <optixu/optixpp_namespace.h>
#include "Denoiser.hpp"

class PTWindow;

class OptixDenoiser : public Denoiser
{
public:
    OptixDenoiser(void);
    ~OptixDenoiser(void) = default;

    void bindBuffers(PTWindow *window) override;
    void resizeBuffers(PTWindow *window) override;
    void denoise(void) override;
    void setBlend(float val) override;

private:
    void setupCommandList(unsigned int width, unsigned int height);

    optix::Context context;
    optix::Buffer primal;  // required
    optix::Buffer normals; // optional
    optix::Buffer albedos; // optional
    optix::Buffer output;
    bool useOptionalFeatures = true;

    optix::CommandList commandListWithDenoiser;
    optix::PostprocessingStage denoiserStage;

    // Amount of original image blended into denoised result, in range [0.0, 1.0]
    float denoiseBlend = 0.0f;
};
