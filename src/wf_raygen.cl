#include "geom.h"
#include "utils.cl"

kernel void genRays(
    global GPUTaskState* tasks,
    global RenderParams* params,
    global QueueCounters* queueLens,
    global uint* raygenQueue,
    global uint* extensionQueue,
    global uint* currPixelIdx,
    uint numTasks
)
{
    // Enqueued with 1D workgroups
    const uint gid_direct = get_global_id(0);
    if (gid_direct >= queueLens->raygenQueue)
        return;

    // Get compacted index
    uint gid = raygenQueue[gid_direct]; // id of path
    uint seed = ReadU32(seed, tasks);
    
    // Calculate pixel coordinates
    uint numPixels = params->width * params->height;
    uint pixelIdx = (*currPixelIdx + gid_direct) % numPixels; // TODO: use gid_local + currentPixelIdx update on host
    WriteU32(pixelIndex, tasks, pixelIdx);

    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction
    float x = (float)(pixelIdx % params->width);
    float y = (float)(pixelIdx / params->width);

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

    // Add paths to extension queue
    uint extIdx = atomic_inc(&queueLens->extensionQueue);
    extensionQueue[extIdx] = gid;

    // TODO: pixel pointer has to be updated on HOST
    // ALSO: reset queue sizes to zero

    WriteU32(seed, tasks, seed);

    // Reset path state
	const float3 zero = (float3)(0.0f);
	const float3 one = (float3)(1.0f);
	WriteFloat3(Ei, tasks, zero);
	WriteFloat3(T, tasks, one);
	WriteU32(pathLen, tasks, 0);
    WriteU32(lastSpecular, tasks, 1);
	WriteF32(lastPdfW, tasks, 1.0f);
    WriteF32(lastPdfDirect, tasks, 0.0f);
    WriteF32(lastPdfImplicit, tasks, 0.0f);
    WriteF32(lastCosTh, tasks, 0.0f);
    WriteF32(lastLightPickProb, tasks, 1.0f);
    WriteF32(shadowRayLen, tasks, 2.0f * params->worldRadius);
    WriteU32(backfaceHit, tasks, 0);
    WriteU32(shadowRayBlocked, tasks, 1);
    WriteFloat3(lastEmission, tasks, zero);
    WriteFloat3(lastBsdf, tasks, zero);
    Hit hit = EMPTY_HIT(FLT_MAX);
    writeHitSoA(hit, tasks, gid, numTasks);
}
