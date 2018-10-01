#include "geom.h"
#include "utils.cl"

// Reset state of all paths. Done after camera/renderparam changes.
kernel void reset(
    global GPUTaskState* tasks,
    global float* pixels,
    global float* denoiserAlbedo,
    global float* denoiserNormal,
    global QueueCounters* queueLens,
    global uint* raygenQueue,
    global RenderParams* params,
    uint numTasks
)
{
	const uint gid = get_global_id(0);

    // Clear screen
	if (gid < params->width * params->height)
    {
        vstore4((float4)(0.0f), gid, pixels);
        vstore4((float4)(0.0f), gid, denoiserNormal);
        // default value for direct emission (not updated in logic kernel)
        vstore4((float4)(0.1f, 0.1f, 0.1f, 0.0f), gid, denoiserAlbedo);
    }
    
    // Clear path data
    if (gid >= numTasks)
		return;

	// Reset path state
	const float4 zero = (float4)(0.0f);
	const float4 one = (float4)(1.0f);
	WriteFloat3(Ei, tasks, zero);
	WriteFloat3(T, tasks, one);
	WriteU32(pathLen, tasks, 0);
	WriteU32(lastSpecular, tasks, 1);
	WriteF32(lastPdfW, tasks, 1.0f);

    // WF params
    WriteF32(lastPdfDirect, tasks, 0.0f);
    WriteF32(lastPdfImplicit, tasks, 0.0f);
    WriteF32(lastCosTh, tasks, 0.0f);
    WriteF32(lastLightPickProb, tasks, 1.0f);
    WriteF32(shadowRayLen, tasks, 2.0f * params->worldRadius);
    WriteU32(backfaceHit, tasks, 0);
    WriteU32(shadowRayBlocked, tasks, 1);
    WriteU32(pixelIndex, tasks, 0);
    WriteU32(firstDiffuseHit, tasks, 0);

    WriteFloat3(lastEmission, tasks, zero);
    WriteFloat3(lastBsdf, tasks, zero);

    // Empty hit
    Hit hit = EMPTY_HIT(FLT_MAX);
    writeHitSoA(hit, tasks, gid, numTasks);

	// Reset RNG seed
    WriteU32(seed, tasks, gid);

    // Put all paths into raygen queue
    raygenQueue[gid] = gid;
    
    if (gid == 0)
        queueLens->raygenQueue = numTasks;
}
