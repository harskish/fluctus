#include "geom.h"
#include "bvh.cl"
#include "utils.cl"

// x and y include offsets when supersampling
kernel void nextVertex(
    global Ray *rays,
    global GPUTaskState *tasks,
    global Material *materials,
    global uchar *texData,
    global TexDescriptor *textures,
    global Triangle *tris,
    global GPUNode *nodes,
    global uint *indices,
    global RenderParams *params,
    global RenderStats *stats,
    uint numTasks)
{
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    if (gid >= limit)
        return;

    // Read the path state
    global GPUTaskState *task = &tasks[gid];
    PathPhase phase = task->phase;
    if (phase != MK_RT_NEXT_VERTEX)
        return;

    Ray r = rays[gid];
    Hit hit = EMPTY_HIT(FLT_MAX); // TODO: Max distance?
    
    bvh_intersect(&r, &hit, tris, nodes, indices);

    // Update render statistics
    // TODO: local atomic_inc within workgroup (check on different hardware)
    atomic_inc(&stats->primaryRays);

    // TEST: show intersection immediately
    if (hit.i > -1)
    {
        float3 N;
        float3 Kd;
        float3 Ks;
        float Ni;
        getMaterialParameters(hit, tris, materials, texData, textures, &Kd, &N, &Ks, &Ni);
        
        task->T = Kd;
        task->pdf = hit.t; // TEST
        task->phase = MK_SPLAT_SAMPLE;
    }
    else
    {
        // Need to clear screen
        task->T = (float3)(0.0f);
        task->pdf = FLT_MAX; // TEST
        task->phase = MK_SPLAT_SAMPLE;
    }
}
