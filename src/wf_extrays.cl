#include "geom.h"
#include "bvh.cl"

// Trace extension ray for all paths in queue
kernel void traceExtension(
    global GPUTaskState* tasks,
    global QueueCounters* queueLens,
    global uint* extQueue,
    global Triangle* tris,
    global GPUNode* nodes,
    global uint* indices,
    global RenderParams* params,
    const uint numTasks
)
{
    uint gid_direct = get_global_id(0);
    if (gid_direct >= queueLens->extensionQueue)
        return;

    const uint gid = extQueue[gid_direct];

    const float3 rayOrig = ReadFloat3(orig, tasks);
    const float3 rayDir = ReadFloat3(dir, tasks);
    Ray r = { rayOrig, rayDir };

    // Trace ray
    Hit hit = EMPTY_HIT(FLT_MAX);
    bvh_intersect(&r, &hit, tris, nodes, indices);
    if (params->sampleImpl && params->useAreaLight) intersectLight(&hit, &r, params);
    
    global uint *len = &ReadU32(pathLen, tasks);
    *len += 1;

    // Write hit to path state
    writeHitSoA(hit, tasks, gid, numTasks);
}