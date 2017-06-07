#include "geom.h"

// Used for interactive preview, usually means camera is moving
// Sample count set to zero to force overwrite immediately after => preview can be biased
kernel void splatPreview(global GPUTaskState *tasks, global float *pixels, global RenderParams *params, uint numTasks)
{
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks);

    if (gid >= limit)
        return;

    // Ignore path state => all threads perform splat
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    // Splat preview
    WriteI32(samples, tasks, 0); // force overwrite on next iteration
    float4 color = (float4)(ReadFloat3(Ei, tasks), 1.0f);
    vstore4(color, (y * params->width + x), pixels);

    // Reset path state
    const float3 zero = (float3)(0.0f);
    const float3 one = (float3)(1.0f);
    WriteFloat3(Ei, tasks, zero);
    WriteFloat3(T, tasks, one);
    WriteF32(pdf, tasks, 1.0f);
    WriteU32(pathLen, tasks, 0);

    // Update phase
    WriteI32(phase, tasks, MK_GENERATE_CAMERA_RAY);
}