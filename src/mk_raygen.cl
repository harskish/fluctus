#include "geom.h"
#include "utils.cl"

// x and y include offsets when supersampling
kernel void genCameraRays(global GPUTaskState *tasks, global RenderParams *params, uint numTasks)
{
    // Enqueued with 1D workgroups
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!
    uint seed = ReadU32(seed, tasks);

    if (gid >= limit)
        return;

    // Read the path state
    global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
    if (*phase != MK_GENERATE_CAMERA_RAY)
        return;
    
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction
    float x = (float)(gid % params->width);
    float y = (float)(gid / params->width);

    // Jittered AA
    x += rand(&seed);
    y += rand(&seed);

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
    float3 rayOrig = params->camera.pos;
    float3 rayTarget = rayOrig + params->camera.right * SCRx + params->camera.up * SCRy + params->camera.dir;
    float3 rayDirection = normalize(rayTarget - rayOrig);

    // Depth of field
    float3 fp = params->camera.pos + rayDirection * params->camera.focalDist;
    float2 rnd = uniformSampleDisk(&seed);
    rayOrig += params->worldRadius * params->camera.apertureSize * (params->camera.right * rnd.x + params->camera.up * rnd.y);
    rayDirection = normalize(fp - rayOrig);

    // Construct camera ray
    WriteFloat3(orig, tasks, rayOrig);
    WriteFloat3(dir, tasks, rayDirection);

    // Update path state
    WriteU32(seed, tasks, seed);
    *phase = MK_RT_NEXT_VERTEX;
}
