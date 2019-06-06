#pragma once

// Denoiser interface, implemented by e.g. OptixDenoiser & BMFRDenoiser

class PTWindow;

class Denoiser
{
public:

    virtual void bindBuffers(PTWindow *window) = 0;
    virtual void resizeBuffers(PTWindow *window) = 0;
    virtual void denoise(void) = 0;

    // Blend denoised result with noisy input
    virtual void setBlend(float val) = 0;
};
