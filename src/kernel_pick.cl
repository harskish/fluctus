#include "geom.h"
#include "bvh.cl"
#include "utils.cl"
#include "intersect.cl"

kernel void pick(global RenderParams *params, global Triangle *tris, global GPUNode *nodes, global uint *indices, global Hit *pickResult, float NDCx, float NDCy)
{
    // Uses one single thread
    if (get_global_id(0) != 0 || get_global_id(1) != 0)
        return;

    // Input coords in NDC-space, [0,1]x[0,1]

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
    float3 rayDir = normalize(rayTarget - params->camera.pos);
     Ray r = { params->camera.pos, rayDir };

    // Trace ray
    Hit hit = EMPTY_HIT(FLT_MAX);
    bvh_intersect(&r, &hit, tris, nodes, indices);
    if (params->sampleImpl && params->useAreaLight) intersectLight(&hit, &r, params);

    // Write result
    *pickResult = hit;
}
