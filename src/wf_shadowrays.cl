#include "geom.h"
#include "bvh.cl"
#include "intersect.cl"

// Trace shadow ray for all paths in queue
kernel void traceShadow(
    global GPUTaskState *tasks,
    global QueueCounters *queueLens,
    global uint *shadowQueue,
    global Triangle *tris,
    global GPUNode *nodes,
    global uint *indices,
    global RenderParams *params,
    uint numTasks
)
{
    uint gid_direct = get_global_id(0);
    if (gid_direct >= queueLens->shadowQueue)
        return;

    const uint gid = shadowQueue[gid_direct];

    const float3 rayOrig = ReadFloat3(shadowOrig, tasks);
    const float3 rayDir = ReadFloat3(shadowDir, tasks);
    Ray r = { rayOrig, rayDir };

    // Trace ray
    float lenL = ReadF32(shadowRayLen, tasks);
    Hit hitL = EMPTY_HIT(lenL);
    
    // TEST: area light not occluding
    if (params->useAreaLight) intersectLight(&hitL, &r, params);
    bool occluded = (hitL.i > -1) || bvh_occluded(&r, &lenL, tris, nodes, indices);

    // Write hit to path state
    WriteU32(shadowRayBlocked, tasks, occluded);

    // Clear queue on HOST
}