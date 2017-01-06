#include "geom.h"
#include "bvh.cl"

// x and y include offsets when supersampling
kernel void nextVertex(global Ray *rays, global GPUTaskState *tasks, global Sphere *spheres, global Triangle *tris, global GPUNode *nodes, global uint *indices, global RenderParams *params, uint numTasks)
{
    const size_t gid = get_global_id(0);
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    // Read the path state
    global GPUTaskState *task = &tasks[gid];
    PathPhase phase = task->phase;
    if (phase != MK_RT_NEXT_VERTEX || gid >= limit)
        return;

    Ray r = rays[gid];
    Hit hit = { (float3)(0.0f), (float3)(0.0f), FLT_MAX, -1 }; // TODO: Max distance?
    bool primWasHit = bvh_intersect_stack(&r, &hit, tris, nodes, indices); // TODO: remove bool return value

    // TEST: show intersection immediately
    if (hit.i > -1)
    {
        task->T = (float3)(1.0f, 0.0f, 0.0f);
        task->pdf = hit.t; // TEST
        task->phase = MK_SPLAT_SAMPLE;
    }
    else
    {
        task->T = (float3)(0.0f, 0.0f, 0.0f);
        task->pdf = FLT_MAX; // TEST
        task->phase = MK_SPLAT_SAMPLE;

        // TODO: gen new ray immediately instead?
        //task->phase = MK_GENERATE_CAMERA_RAY;
    }
}
