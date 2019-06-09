#pragma once

#include "Denoiser.hpp"
#include <clt.hpp>

class PTWindow;
class CLContext;

/*
    'Blockwise Multi-Order Feature Regression for Real-Time Path Tracing Reconstruction' by Koskela et al.
    
    A realtime (~2ms @ 720p) block-based denoiser with built in TAA.
    Based on the reference implementation: https://github.com/maZZZu/bmfr
*/

// Creates two same buffers and swap() call can be used to change which one is considered
// current and which one previous
template <class T>
class Double_buffer
{
private:
    T a, b;
    bool swapped;

public:
    template <typename... Args>
    Double_buffer(Args... args) : a(args...), b(args...), swapped(false) {};
    T *current() { return swapped ? &a : &b; }
    T *previous() { return swapped ? &b : &a; }
    void swap() { swapped = !swapped; }
};

//class BMFRKernelBase;
//class FitterKernel;

class BMFRDenoiser : public Denoiser
{
    friend class BMFRKernelBase;
    friend class FitterKernel;
    friend class WeightedSumKernel;
    friend class AccumNoisyKernel;
    friend class AccumFilteredKernel;
    friend class TAAKernel;

public:
    BMFRDenoiser(void) = default;
    ~BMFRDenoiser(void) = default;

    void bindBuffers(PTWindow *window) override;
    void resizeBuffers(PTWindow *window) override;
    void denoise(void) override;
    void setBlend(float val) override;

    void setup(CLContext* ctx, PTWindow* window);

private:

    int buffer_count = 0;
    int features_not_scaled_count = 0;
    int features_scaled_count = 0;

    // Buffers
    Double_buffer<cl::Buffer> normals_buffer;
    Double_buffer<cl::Buffer> positions_buffer;
    Double_buffer<cl::Buffer> noisy_buffer;
    cl::Buffer in_buffer;
    cl::Buffer filtered_buffer;
    Double_buffer<cl::Buffer> out_buffer;
    Double_buffer<cl::Buffer> result_buffer;
    cl::Buffer prev_pixels_buffer;
    cl::Buffer accept_buffer;
    cl::Buffer albedo_buffer;
    cl::Buffer tone_mapped_buffer;
    cl::Buffer weights_buffer;
    cl::Buffer mins_maxs_buffer;
    Double_buffer<cl::Buffer> spp_buffer;
    std::vector<Double_buffer<cl::Buffer> *> all_double_buffers;

    // Kernels
    clt::Kernel* fitter_kernel;
    clt::Kernel* weighted_sum_kernel;
    clt::Kernel* accum_noisy_kernel;
    clt::Kernel* accum_filtered_kernel;
    clt::Kernel* taa_kernel;

    // Amount of original image blended into denoised result, in range [0.0, 1.0]
    float denoiseBlend = 0.0f;
    CLContext* ctx;

    unsigned int frame = 0;
};
