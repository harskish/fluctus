#include "geom.h"

// x and y include offsets when supersampling
kernel void splat(global Ray *rays, global GPUTaskState *tasks, read_only image2d_t src, write_only image2d_t dst, global RenderParams *params, uint numTasks)
{
    //const size_t gid = get_global_id(0);
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    if (gid >= limit)
        return;

    // Read the path state
    global GPUTaskState *task = &tasks[gid];
    PathPhase phase = task->phase;
    if (phase != MK_SPLAT_SAMPLE)
        return;

    //const uint x = gid % params->width;
    //const uint y = gid / params->width;
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    // TEST: just show throughput as color
    const float4 color = (float4)(task->T, 0.0f);
    write_imagef(dst, (int2)(x, y), color);

    // Update phase
    task->phase = MK_GENERATE_CAMERA_RAY;
}
