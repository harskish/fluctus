#include "geom.h"

// x and y include offsets when supersampling
kernel void splat(global GPUTaskState *tasks, global float *pixels, global RenderParams *params, global RenderStats *stats, uint numTasks, uint iteration)
{
    //const size_t gid = get_global_id(0);
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    if (gid >= limit)
        return;

    // Read the path state
    global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
    if (*phase != MK_SPLAT_SAMPLE)
        return;

	// Accumulate radiance
	global int *samples = (global int*)&ReadI32(samples, tasks);
    float4 color = (float4)(ReadFloat3(Ei, tasks), 1.0f);
	float4 prev = vload4(gid, pixels);
	if (*samples > 0) color += prev;
	vstore4(color, gid, pixels);
	*samples += 1;
    atomic_inc(&stats->samples);

	// Reset path state
	const float3 zero = (float3)(0.0f);
	const float3 one = (float3)(1.0f);
	WriteFloat3(Ei, tasks, zero);
	WriteFloat3(T, tasks, one);
	WriteU32(pathLen, tasks, 0);

	// Just keep accumulating seed
    //uint seed = get_global_id(1) * params->width + get_global_id(0) + *samples * params->width * params->height; // unique for each pixel
    //WriteU32(seed, tasks, seed);
    // TODO: more

    // Update phase
    *phase = MK_GENERATE_CAMERA_RAY;
}
