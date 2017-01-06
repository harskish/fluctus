#include "geom.h"

// x and y include offsets when supersampling
kernel void genCameraRays(global Ray *rays, global GPUTaskState *tasks, global RenderParams *params, uint numTasks)
{
    // Enqueued with 1D workgroups
    const size_t gid = get_global_id(0);
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    // Read the path state
    global GPUTaskState *task = &tasks[gid];
    PathPhase phase = task->phase;
    if (phase != MK_GENERATE_CAMERA_RAY || gid >= limit)
        return;

    
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction
    float x = (float)(gid % params->width);
    float y = (float)(gid / params->width);

    // NDC-space, [0,1]x[0,1]
    float NDCx = x / params->width;
    float NDCy = y / params->height;

    // Screen space, [-1,1]x[-1,1]
    float SCRx = 2.0f * NDCx - 1.0f;
    float SCRy = 2.0f * NDCy - 1.0f;

    // Aspect ratio fix applied horizontally
    SCRx *= (float)params->width / params->height;

    // Screen space coordinates scaled based on fov
    float scale = tan(toRad(0.5f * params->camera.fov)); // half of width
    SCRx *= scale;
    SCRy *= scale;

    // World space coorinates of pixel
    float3 rayTarget = params->camera.pos + params->camera.right * SCRx + params->camera.up * SCRy + params->camera.dir;
    float3 rayDirection = normalize(rayTarget - params->camera.pos);

    // Construct camera ray
    Ray r = { params->camera.pos, rayDirection };
    rays[gid] = r;

    // Update phase
    task->phase = MK_RT_NEXT_VERTEX;
}
