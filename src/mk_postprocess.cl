#include "geom.h"
#include "tonemap.cl"


// Performs post processing, e.g. tone mapping
// Restult is shown on the screen by OpenGL or written into a file for exporting
kernel void process(
    global float *pixelsRaw,      // CL only
    global float *denoiserAlbedo, // CL only
    global float *denoiserNormal, // CL only
    global float *pixelsPreview,    // GL-CL shared
    global float *denoiserAlbedoGL, // GL-CL shared
    global float *denoiserNormalGL, // GL-CL shared
    global RenderParams *params,
    uint numTasks)
{
    // PixelPreview is as big as the render resolution
    // PixelsRaw is as big as numTasks

    const size_t gid = get_global_id(0);
    const uint limit = params->width * params->height;
    if (gid >= limit)
        return;

	float4 color = vload4(gid, pixelsRaw);
    
    PostProcessParams par = params->ppParams;

    // Divide accumulated radiance with number of samples
    if (color.w > 0.0)
        color = color / color.w;

    // Exposure adjustment
    color.xyz *= par.exposure;

    // Tonemapping (indices in tracer_ui.cpp)
    if (par.tmOperator == 1)
        color.xyz = reinhardTonemap(color.xyz);
    if (par.tmOperator == 2)
        color.xyz = uncharted2Tonemap(color.xyz);

    // Gamma correction
    color.xyz = pow(color.xyz, 1.0f / 2.2f);
    
    // Output color
    vstore4(color, gid, pixelsPreview);

    // Output optional denoiser features
#ifdef USE_OPTIX_DENOISER
    float4 normal = vload4(gid, denoiserNormal);
    vstore4((normal.w > 1.0f) ? normal / normal.w : normal, gid, denoiserNormalGL);
    float4 albedo = vload4(gid, denoiserAlbedo);
    vstore4((albedo.w > 1.0f) ? albedo / albedo.w : albedo, gid, denoiserAlbedoGL);
#endif
}
